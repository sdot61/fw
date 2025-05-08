import re
import string

from flask import Flask, request, render_template

from rapidfuzz import process, fuzz
import jellyfish
import Levenshtein

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False   # turn off in prod

# --- Pre-load the text & build indexes --------------------------

with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

# vocabulary: lowercase stripped words → set of originals & positions
vocab = set()
positions = {}  # word_lower → [ { "line": int, "word": original } ]
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


# --- Flask routes ----------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    # On GET, just render empty form
    if request.method == "GET":
        return render_template("index.html", match=[], search_word="")

    # On POST, pull the form field
    q = request.form.get("searchWord", "").strip().lower()
    if not q:
        return render_template("index.html", match=[], search_word="")

    # 1) phonetic “cognates” via Metaphone
    code = jellyfish.metaphone(q)
    phonetic_matches = phonetic_buckets.get(code, [])

    # 2) close by edit-distance (<= 3 edits)
    lev_matches = [w for w in vocab if Levenshtein.distance(q, w) <= 3]

    # 3) fuzzy match (token-sort ratio)
    fuzzy_results = process.extract(
        q,
        vocab,
        scorer=fuzz.token_sort_ratio,
        limit=100
    )
    fuzzy_matches = [w for w, score, _ in fuzzy_results if score >= 60]

    # Combine all signals
    all_matches = set(phonetic_matches) | set(lev_matches) | set(fuzzy_matches)

    # Build the context for template
    out = []
    for w in all_matches:
        out.append({
            "match": w,
            "positions": positions.get(w, [])
        })

    return render_template(
        "index.html"
