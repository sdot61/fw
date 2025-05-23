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

# --- Load & index the Wake -----------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}  # word → list of {word, line}
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

# --- n-gram overlap helper -----------------
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

    # precompute cleaned vocab forms
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}

    # 1) literal-exact bypass: any w with cleaned[w]==q_clean
    #    sorted by first appearance in text
    exact = [
        w for w, w_cl in cleaned.items()
        if w_cl == q_clean
    ]
    exact.sort(key=lambda w: min(p["line"] for p in positions[w]))

    # 2) Build scores for everything else
    scores = {}
    def boost(w, sc):
        scores[w] = max(scores.get(w, 0), sc)

    # A) substring boost → 100
    for w, w_cl in cleaned.items():
        if w in exact:
            continue
        if q_clean in w_cl:
            boost(w, 100)

    # B) prefix bump → +10 if already substring-matched
    for w, w_cl in cleaned.items():
        if w in exact:
            continue
        if scores.get(w, 0) > 0 and w_cl.startswith(q_clean):
            scores[w] += 10

    # C) Jaro–Winkler ≥ .80 → up to 100
    for w, w_cl in cleaned.items():
        if w in exact:
            continue
        jw = jellyfish.jaro_winkler_similarity(q_clean, w_cl)
        if jw >= 0.80:
            boost(w, int(jw * 100))

    # D) trigram-overlap ≥ .6 → up to 100
    for w, w_cl in cleaned.items():
        if w in exact:
            continue
        ov = ngram_overlap(q_clean, w_cl, n=3)
        if ov >= 0.6:
            boost(w, int(ov * 100))

    # E) fuzzy token_set_ratio ≥ 70 → up to 100
    for w, w_cl in cleaned.items():
        if w in exact:
            continue
        ts = fuzz.token_set_ratio(q_clean, w_cl)
        if ts >= 70:
            boost(w, ts)

    # F) fuzzy partial_ratio ≥ 70 → up to 100
    for w, w_cl in cleaned.items():
        if w in exact:
            continue
        pr = fuzz.partial_ratio(q_clean, w_cl)
        if pr >= 70:
            boost(w, pr)

    # G) raw token_sort_ratio ≥ 60 → up to 100
    for w, sc, _ in process.extract(
        q, vocab, scorer=fuzz.token_sort_ratio, limit=200
    ):
        if w in exact:
            continue
        if sc >= 60:
            boost(w, sc)

    # H) raw partial_ratio ≥ 60 → up to 100
    for w, sc, _ in process.extract(
        q, vocab, scorer=fuzz.partial_ratio, limit=200
    ):
        if w in exact:
            continue
        if sc >= 60:
            boost(w, sc)

    # I) Levenshtein distance → up to 100
    thresh = 2 if len(q_clean) <= 5 else 3
    for w in vocab:
        if w in exact:
            continue
        if abs(len(w) - len(q)) <= thresh:
            d = Levenshtein.distance(q, w)
            if d <= thresh:
                boost(w, 100 - d * 10)

    # J) exact Double-Metaphone → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if not code:
            continue
        for w in phonetic_buckets.get(code, []):
            if w not in exact:
                boost(w, 95)

    # K) short-word penalty (len ≤4, non-exact)
    for w in list(scores):
        if cleaned[w] != q_clean:
            L = len(w)
            if L <= 4:
                penalty = (5 - L) * 10  # 4→10,3→20,2→30,1→40
                scores[w] = max(0, scores[w] - penalty)

    # L) sort by (score DESC, length DESC)
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], -len(kv[0]))
    )

    # M) tail-demotion: 1–2 letters or over-common 3s
    tail = {
        w for w, _ in ranked
        if len(w) <= 2
           or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail]
    tail_list = [w for w, _ in ranked if w in tail]
    scored_results = primary + tail_list

    # N) final filter: drop single-letter & non-ASCII (if multi-char query)
    combined = exact + scored_results
    if len(q_clean) > 1:
        combined = [
            w for w in combined
            if len(cleaned[w]) > 1 and w.isascii()
        ]

    return combined[:max_results]

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
