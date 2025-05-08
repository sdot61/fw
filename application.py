import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import jellyfish
import Levenshtein

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

phonetic_buckets = {}
for w in vocab:
    code = jellyfish.metaphone(w)
    phonetic_buckets.setdefault(code, []).append(w)


# --- Flask routes -----------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_word = request.form.get("searchWord", "").strip().lower()
        if not search_word:
            return render_template("index.html", match=[], search_word="")

        # phonetic matches
        qc = jellyfish.metaphone(search_word)
        phonetic_matches = phonetic_buckets.get(qc, [])

        # edit-distance matches
        lev_matches = [
            w for w in vocab
            if Levenshtein.distance(search_word, w) <= 3
        ]

        # fuzzy-ratio matches
        fuzzy_results = process.extract(
            search_word,
            vocab,
            scorer=fuzz.token_sort_ratio,
            limit=100
        )
        fuzzy_matches = [w for w, score, _ in fuzzy_results if score >= 60]

        # combine
        all_matches = set(phonetic_matches) | set(lev_matches) | set(fuzzy_matches)
        out = [
            {"match": w, "positions": positions.get(w, [])}
            for w in all_matches
        ]

        return render_template("index.html", match=out, search_word=search_word)

    # GET
    return render_template("index.html", match=[], search_word="")


@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True, silent=True) or {}
    q = data.get("query", "").strip().lower()
    if not q:
        return jsonify([])

    # phonetic
    qc = jellyfish.metaphone(q)
    phonetic_matches = phonetic_buckets.get(qc, [])

    # edit-distance
    lev_matches = [
        w for w in vocab
        if Levenshtein.distance(q, w) <= 3
    ]

    # fuzzy
    fuzzy_results = process.extract(
        q,
        vocab,
        scorer=fuzz.token_sort_ratio,
        limit=100
    )
    fuzzy_matches = [w for w, score, _ in fuzzy_results if score >= 60]

    all_matches = set(phonetic_matches) | set(lev_matches) | set(fuzzy_matches)
    out = [
        {"match": w, "positions": positions.get(w, [])}
        for w in all_matches
    ]
    return jsonify(out)


if __name__ == "__main__":
    # when run directly, listen on 8080
    app.run(host="0.0.0.0", port=8080)
