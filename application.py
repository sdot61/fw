import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import fuzz, process
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration constants -------------------------
High_Freq_Cutoff    = 6
DEFAULT_MAX_RESULTS = 700

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False

# --- Load the text & build indexes -------------------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}  # word_lower → list of { line: int, word: original }
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

# --- Build phonetic buckets via Double Metaphone -----------
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- Utility Functions --------------------------------------
def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

def syllabic_overlap(query: str, word: str) -> float:
    query_chunks = re.findall(r"[aeiouy]+[^aeiouy\s]*", query.lower())
    word_chunks = re.findall(r"[aeiouy]+[^aeiouy\s]*", word.lower())
    if not query_chunks or not word_chunks:
        return 0.0
    return len(set(query_chunks) & set(word_chunks)) / len(query_chunks)

def length_penalty(word: str, base_score: float) -> float:
    len_score = max(0.7, min(1.0, 8.0 / max(len(word), 1)))
    return base_score * len_score

# --- Core matching function -------------------------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    def boost(w, raw_score):
        score = length_penalty(w, raw_score)
        scores[w] = max(scores.get(w, 0), score)

    # STEP 1: cleaned-prefix boost → 100
    prefix_len = min(4, len(q_clean))
    if prefix_len > 0:
        for w in vocab:
            if re.sub(r"[^a-z0-9]", "", w).startswith(q_clean[:prefix_len]):
                boost(w, 100)

    # STEP 2: substring match → 95
    for w in vocab:
        if q_clean in re.sub(r"[^a-z0-9]", "", w):
            boost(w, 95)

    # STEP 3: fuzzy cleaned matches
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        ts = fuzz.token_sort_ratio(q_clean, w_clean)
        pr = fuzz.partial_ratio(q_clean, w_clean)
        if ts >= 70:
            boost(w, ts)
        if pr >= 70:
            boost(w, pr)

    # STEP 4: raw fuzzy matches
    for scorer, threshold in [(fuzz.token_sort_ratio, 60), (fuzz.partial_ratio, 60)]:
        for w, sc, _ in process.extract(q, vocab, scorer=scorer, limit=200):
            if sc >= threshold:
                boost(w, sc)

    # STEP 5: bigram-overlap ≥ 0.5
    for w in vocab:
        ov = ngram_overlap(q, w)
        if ov >= 0.5:
            boost(w, ov * 100)

    # STEP 6: Levenshtein distance
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                boost(w, 100 - (10 * d))

    # STEP 7: Double-Metaphone phonetic matches
    for code in doublemetaphone(q_clean):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 100)

    # STEP 8: Syllabic overlap
    for w in vocab:
        so = syllabic_overlap(q, w)
        if so > 0.4:
            boost(w, so * 100)

    # FINAL RANKING: score descending
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])

    return [w for w, _ in ranked][:max_results]

# --- Flask routes -----------------------------------------
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
