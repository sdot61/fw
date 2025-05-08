import re
import string

from flask import Flask, request, jsonify, render_template

# fuzzy and distance libs
from rapidfuzz import process, fuzz
import jellyfish
import Levenshtein

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # turn off in prod

# ---  Pre‐load the text & build indexes  --------------------------

with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

# vocabulary: lowercase stripped words → list of appearances
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

# Precompute phonetic (Metaphone) buckets
phonetic_buckets = {}
for w in vocab:
    code = jellyfish.metaphone(w)
    phonetic_buckets.setdefault(code, []).append(w)


# ---  Flask routes  ------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
    q = query.strip().lower()
    if not q:
        return jsonify([])

    # 1) phonetic “cognates” via Metaphone
    qc = jellyfish.metaphone(q)
    phonetic_matches = phonetic_buckets.get(qc, [])

    # 2) close by edit‐distance (<= 3 edits)
    lev_matches = [
        w for w in vocab
        if Levenshtein.distance(q, w) <= 3
    ]

    # 3) fuzzy ratio (token‐sort for word order invariance)
    fuzzy_results = process.extract(
        q,
        vocab,
        scorer=fuzz.token_sort_ratio,
        limit=100
    )
    fuzzy_matches = [w for w, score, _ in fuzzy_results if score >= 60]

    # Combine all signals
    all_matches = set(phonetic_matches) \
                | set(lev_matches) \
                | set(fuzzy_matches)

    # Build the JSON response
    out = []
    for w in all_matches:
        out.append({
            "match": w,
            "positions": positions.get(w, [])
        })

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
