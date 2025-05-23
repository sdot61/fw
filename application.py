import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
import jellyfish
from doublemetaphone import doublemetaphone

# --- Configuration -------------------------
High_Freq_Cutoff    = 6     # demote 3-letter words occurring ≥ this
DEFAULT_MAX_RESULTS = 700   # cap on total results

# --- Flask setup ---------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # off in prod

# --- Load & index the Wake -----------------
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
        positions.setdefault(w, []).append({"word": raw, "line": lineno})

# --- Build phonetic buckets ---------------
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- n-gram overlap helper ----------------
def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

# --- Core matching pipeline ----------------
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

    # 1) full cleaned-substring → 100
    for w, w_cl in cleaned.items():
        if q_clean in w_cl:
            boost(w, 100)

    # 2) light prefix bump → +10
    for w, w_cl in cleaned.items():
        if scores.get(w, 0) > 0 and w_cl.startswith(q_clean):
            scores[w] += 10

    # 3) suffix boost → 95 (e.g. "-pest" → "judapest")
    suffix_len = min(len(q_clean), max(4, len(q_clean)//2))
    suffix = q_clean[-suffix_len:]
    for w, w_cl in cleaned.items():
        if w_cl.endswith(suffix):
            boost(w, 95)

    # 4) Jaro–Winkler ≥ .80 → up to 100
    for w, w_cl in cleaned.items():
        jw = jellyfish.jaro_winkler_similarity(q_clean, w_cl)
        if jw >= 0.80:
            boost(w, int(jw * 100))

    # 5) trigram-overlap ≥ .60 → up to 100
    for w, w_cl in cleaned.items():
        ov = ngram_overlap(q_clean, w_cl)
        if ov >= 0.60:
            boost(w, int(ov * 100))

    # helper for coverage factor
    def coverage_factor(w_cl: str) -> float:
        cov = len(w_cl) / len(q_clean)
        # scale linearly: below .5 → factor = cov/.5, above → 1
        return cov / 0.5 if cov < 0.5 else 1.0

    # 6) cleaned token_set_ratio ≥ 70 → up to 100, penalized by coverage
    for w, w_cl in cleaned.items():
        ts = fuzz.token_set_ratio(q_clean, w_cl)
        if ts >= 70:
            factor = coverage_factor(w_cl)
            boost(w, int(ts * factor))

    # 7) cleaned partial_ratio ≥ 70 → up to 100, penalized by coverage
    for w, w_cl in cleaned.items():
        pr = fuzz.partial_ratio(q_clean, w_cl)
        if pr >= 70:
            factor = coverage_factor(w_cl)
            boost(w, int(pr * factor))

    # 8) raw fuzzy token_sort_ratio ≥ 60, penalized by coverage
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        w_cl = cleaned[w]
        if sc >= 60:
            factor = coverage_factor(w_cl)
            boost(w, int(sc * factor))

    # 9) raw fuzzy partial_ratio ≥ 60, penalized by coverage
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        w_cl = cleaned[w]
        if sc >= 60:
            factor = coverage_factor(w_cl)
            boost(w, int(sc * factor))

    # 10) Levenshtein distance (≤2 edits if short, ≤3 if long)
    thresh = 2 if len(q_clean) <= 5 else 3
    for w in vocab:
        w_cl = cleaned[w]
        if abs(len(w_cl) - len(q_clean)) <= thresh:
            d = Levenshtein.distance(q_clean, w_cl)
            if d <= thresh:
                boost(w, 100 - (d * 10))

    # 11) exact phonetic match → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # 12) penalty for short (≤5) non-exact matches
    for w in list(scores):
        L = len(cleaned[w])
        if L <= 5 and cleaned[w] != q_clean:
            penalty = (6 - L) * 10  # 5→10,4→20,3→30,2→40
            scores[w] = max(0, scores[w] - penalty)

    # Final sort: score desc, then length desc
    ranked = sorted(scores.items(),
                    key=lambda kv: (-kv[1], -len(kv[0])))

    # Tail-demotion: 1–2 letters or over-common 3 letters
    tail_set = {
        w for w, _ in ranked
        if len(cleaned[w]) <= 2
           or (len(cleaned[w]) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail_set]
    tail    = [w for w, _ in ranked if w in tail_set]
    ordered = primary + tail

    # Drop single-letter & non-ASCII tokens for multi-char queries
    if len(q_clean) > 1:
        ordered = [w for w in ordered
                   if len(cleaned[w]) > 1 and w.isascii()]

    return ordered[:max_results]


# --- Flask routes ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        term = request.form.get("searchWord", "").strip()
        if not term:
            return render_template("index.html", match=[], search_word="")
        ms = find_matches(term, vocab, phonetic_buckets)
        out = [{"match": w, "positions": positions[w]} for w in ms]
        return render_template("index.html", match=out, search_word=term)
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
