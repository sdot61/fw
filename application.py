import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
from doublemetaphone import doublemetaphone

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # turn off in prod

# --- Pre-load the text & build indexes -------------------------

with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}  # word_lower → [ { line: int, word: original} ]
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
    primary, secondary = doublemetaphone(w)
    for code in (primary, secondary):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)


# --- Matching utility -----------------------------------------

def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))


def find_matches(query, vocab, phonetic_buckets, max_results=500):
    q = query.lower()
    scores = {}

    # 1) substring boost
    for w in vocab:
        if q in w:
            scores[w] = max(scores.get(w, 0), 100)

    # 2) fuzzy token_sort_ratio
    for w, score, _ in process.extract(q, vocab, scorer=fuzz.token_sort_ratio, limit=200):
        if score >= 50:
            scores[w] = max(scores.get(w, 0), score)

    # 3) fuzzy partial_ratio (for embedded matches)
    for w, score, _ in process.extract(q, vocab, scorer=fuzz.partial_ratio, limit=200):
        if score >= 50:
            scores[w] = max(scores.get(w, 0), score)

    # 4) Levenshtein distance
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                lev_score = 100 - (d * 10)
                scores[w] = max(scores.get(w, 0), lev_score)

    # 5) bigram overlap
    for w in vocab:
        ov = ngram_overlap(q, w, n=2)
        if ov >= 0.5:
            scores[w] = max(scores.get(w, 0), int(ov * 100))

    # 6) phonetic match via Double Metaphone
    pcode, scode = doublemetaphone(q)
    for code in (pcode, scode):
        if not code:
            continue
        for w in phonetic_buckets.get(code, []):
            scores[w] = max(scores.get(w, 0), 80)

    # rank all scored items descending by score
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    # drop any word of length ≤ 2
    filtered = [(w, sc) for w, sc in ranked if len(w) > 2]
    # separate into >3 letters and exactly 3 letters
    longer       = [w for w, sc in filtered if len(w) > 3]
    three_letter = [w for w, sc in filtered if len(w) == 3]
    # within 3-letter group, split high-frequency (positions >10) 
    low_freq_short  = [w for w in three_letter if len(positions.get(w, [])) <= 10]
    high_freq_short = [w for w in three_letter if len(positions.get(w, [])) > 10]
    # concatenate: longer first, then low-frequency 3-letter, then high-frequency 3-letter
    ordered = longer + low_freq_short + high_freq_short
    # return top max_results
    return ordered[:max_results]


# --- Flask routes -----------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_word = request.form.get("searchWord", "").strip().lower()
        if not search_word:
            return render_template("index.html", match=[], search_word="")

        matches = find_matches(search_word, vocab, phonetic_buckets)
        out = [
            {"match": w, "positions": positions.get(w, [])}
            for w in matches
        ]

        return render_template("index.html", match=out, search_word=search_word)

    return render_template("index.html", match=[], search_word="")


@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True, silent=True) or {}
    q = data.get("query", "").strip().lower()
    if not q:
        return jsonify([])

    matches = find_matches(q, vocab, phonetic_buckets)
    out = [
        {"match": w, "positions": positions.get(w, [])}
        for w in matches
    ]
    return jsonify(out)


@app.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html", lines=enumerate(lines))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
