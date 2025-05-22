import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration constants -------------------------
High_Freq_Cutoff    = 6    # any 1–3 letter word occurring ≥ this will be demoted
DEFAULT_MAX_RESULTS = 700  # cap on number of results

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # turn off in prod

# --- Pre-load the text & build indexes -------------------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}  # word_lower → [ { line: int, word: original } ]
word_re = re.compile(r"\b[\w'-]+\b")

for lineno, line in enumerate(lines):
    for raw in word_re.findall(line):
        stripped = raw.strip(string.punctuation)
        if not stripped:
            continue
        lower = stripped.lower()
        vocab.add(lower)
        positions.setdefault(lower, []).append({
            "word": raw,
            "line": lineno
        })

# Build phonetic buckets using Double Metaphone (primary + secondary)
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)


# --- Matching utility -----------------------------------------
def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))


def find_matches(query, vocab, phonetic_buckets, max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    scores = {}

    # … your existing substring, fuzzy, Levenshtein, bigram logic (1–5) …

    # 6a) exact phonetic matches → boost to 100
    pcode, scode = doublemetaphone(q)
    if pcode:
        for w in phonetic_buckets.get(pcode, []):
            scores[w] = max(scores.get(w, 0), 100)
    if scode:
        for w in phonetic_buckets.get(scode, []):
            scores[w] = max(scores.get(w, 0), 100)

    # 6b) soft phonetic matches (edit-distance ≤ 2) → boost to 90
    for bucket_code, words in phonetic_buckets.items():
        if pcode and Levenshtein.distance(pcode, bucket_code) <= 2:
            for w in words:
                scores[w] = max(scores.get(w, 0), 90)
        elif scode and Levenshtein.distance(scode, bucket_code) <= 2:
            for w in words:
                scores[w] = max(scores.get(w, 0), 90)

    # 7) rank by score descending
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])

    # 8) identify the “tail” words:
    #    • All 1–2 letter words, plus
    #    • 3-letter words with freq ≥ High_Freq_Cutoff
    tail_set = {
        w
        for w, _ in ranked
        if len(w) <= 2
           or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }

    # 9) primary: everything except those in tail_set
    primary = [w for w, _ in ranked if w not in tail_set]

    # 10) tail: only those demoted by rule
    tail = [w for w, _ in ranked if w in tail_set]

    # 11) final ordering + cap
    ordered = primary + tail
    return ordered[:max_results]


# --- Flask routes -----------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        q = request.form.get("searchWord", "").strip().lower()
        if not q:
            return render_template("index.html", match=[], search_word="")

        matches = find_matches(q, vocab, phonetic_buckets)
        out = [{"match": w, "positions": positions[w]} for w in matches]
        return render_template("index.html", match=out, search_word=q)

    return render_template("index.html", match=[], search_word="")


@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True, silent=True) or {}
    q = data.get("query", "").strip().lower()
    if not q:
        return jsonify([])
    matches = find_matches(q, vocab, phonetic_buckets)
    return jsonify([{"match": w, "positions": positions[w]} for w in matches])


@app.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html", lines=enumerate(lines))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
