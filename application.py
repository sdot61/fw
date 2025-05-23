import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
import jellyfish
from doublemetaphone import doublemetaphone

# --- Configuration -------------------------
High_Freq_Cutoff    = 6     # demote very common 3-letter words
DEFAULT_MAX_RESULTS = 700   # cap on total results

# --- Flask setup ---------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # off in prod

# --- Load & index Finnegans Wake ------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}
word_re = re.compile(r"\b[\w'-]+\b")
for lineno, line in enumerate(lines):
    for raw in word_re.findall(line):
        stripped = raw.strip(string.punctuation)
        if not stripped:
            continue
        w = stripped.lower()
        vocab.add(w)
        positions.setdefault(w, []).append({
            "word": raw,
            "line": lineno
        })

# --- Build phonetic buckets via Double Metaphone ---
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- Helpers ---------------------------------
def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

def coverage_factor(w_cl: str, q_cl: str) -> float:
    cov = len(w_cl) / len(q_cl)
    if cov < 0.5:
        return cov / 0.5
    if cov > 1.0:
        # slight boost for longer words
        return 1.0 + (cov - 1.0) * 0.5
    return 1.0

# --- Core matching pipeline -----------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower().strip()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    if not q_clean:
        return []

    # precompute cleaned vocab once
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}
    scores = {}

    def boost(w, sc):
        scores[w] = max(scores.get(w, 0), sc)

    # STEP 0: exact-clean match → 10_000
    for w, w_cl in cleaned.items():
        if w_cl == q_clean:
            boost(w, 10_000)

    # STEP 1: substring boost → 100
    for w, w_cl in cleaned.items():
        if q_clean in w_cl:
            boost(w, 100)

    # STEP 2: prefix boost → 110
    for w, w_cl in cleaned.items():
        if w_cl.startswith(q_clean):
            boost(w, 110)

    # STEP 3: suffix boost → 85
    suffix_len = min(len(q_clean), max(4, len(q_clean)//2))
    suffix = q_clean[-suffix_len:]
    for w, w_cl in cleaned.items():
        if w_cl.endswith(suffix):
            boost(w, 85)

    # STEP 4: Jaro–Winkler ≥ .80 → up to 100
    for w, w_cl in cleaned.items():
        jw = jellyfish.jaro_winkler_similarity(q_clean, w_cl)
        if jw >= 0.80:
            boost(w, int(jw * 100))

    # STEP 5: trigram overlap ≥ .60 → up to 100
    for w, w_cl in cleaned.items():
        ov = ngram_overlap(q_clean, w_cl, n=3)
        if ov >= 0.6:
            boost(w, int(ov * 100))

    # STEP 6: token_set_ratio ≥ 70 → up to 100, coverage-weighted
    for w, w_cl in cleaned.items():
        ts = fuzz.token_set_ratio(q_clean, w_cl)
        if ts >= 70:
            factor = coverage_factor(w_cl, q_clean)
            boost(w, int(ts * factor))

    # STEP 7: partial_ratio ≥ 70 → up to 100, coverage-weighted
    for w, w_cl in cleaned.items():
        pr = fuzz.partial_ratio(q_clean, w_cl)
        if pr >= 70:
            factor = coverage_factor(w_cl, q_clean)
            boost(w, int(pr * factor))

    # STEP 8: raw token_sort_ratio ≥ 60 → up to 100, coverage-weighted
    for w, sc, _ in process.extract(
        q, vocab, scorer=fuzz.token_sort_ratio, limit=200
    ):
        w_cl = cleaned[w]
        if sc >= 60:
            boost(w, sc)
            factor = coverage_factor(w_cl, q_clean)
            boost(w, int(sc * factor))

    # STEP 9: raw partial_ratio ≥ 60 → up to 100, coverage-weighted
    for w, sc, _ in process.extract(
        q, vocab, scorer=fuzz.partial_ratio, limit=200
    ):
        w_cl = cleaned[w]
        if sc >= 60:
            boost(w, sc)
            factor = coverage_factor(w_cl, q_clean)
            boost(w, int(sc * factor))

    # STEP 10: Levenshtein distance (≤2 if short, ≤3 if long)
    thresh = 2 if len(q_clean) <= 5 else 3
    for w in vocab:
        w_cl = cleaned[w]
        if abs(len(w_cl) - len(q_clean)) <= thresh:
            d = Levenshtein.distance(q_clean, w_cl)
            if d <= thresh:
                boost(w, 100 - d * 10)

    # STEP 11: exact Double-Metaphone → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # --- Final sort: by score desc, then length desc
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], -len(kv[0]))
    )

    # --- Tail-demotion: 1–2 letters & over-common 3's
    tail_set = {
        w for w, _ in ranked
        if len(w) <= 2
           or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail_set]
    tail    = [w for w, _ in ranked if w in tail_set]
    ordered = primary + tail

    # --- Filter out single-letter & non-ASCII tokens (multi-char queries)
    if len(q_clean) > 1:
        ordered = [
            w for w in ordered
            if len(cleaned[w]) > 1 and w.isascii()
        ]

    return ordered[:max_results]

# --- Flask routes ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        q = request.form.get("searchWord", "").strip()
        if not q:
            return render_template("index.html", match=[], search_word="")
        ms = find_matches(q, vocab, phonetic_buckets)
        out = [{"match": w, "positions": positions[w]} for w in ms]
        return render_template("index.html", match=out, search_word=q)
    return render_template("index.html", match=[], search_word="")

@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True) or {}
    q = data.get("query", "").strip()
    if not q:
        return jsonify([])
    ms = find_matches(q, vocab, phonetic_buckets)
    return jsonify([{"match": w, "positions": positions[w]} for w in ms])

@app.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html", lines=enumerate(lines))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
