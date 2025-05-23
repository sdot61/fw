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

# --- Core matching function -------------------------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    def boost(w, score):
        scores[w] = max(scores.get(w, 0), score)

    # STEP 1: exact, prefix, or substring match
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        if q_clean == w_clean:
            boost(w, 100)
        elif w_clean.startswith(q_clean):
            boost(w, 95)
        elif q_clean in w_clean:
            boost(w, 90)
        elif w_clean in q_clean:
            boost(w, 85)

    # STEP 2: phonetic match
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # STEP 3: fuzzy match
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        ts = fuzz.token_sort_ratio(q_clean, w_clean)
        pr = fuzz.partial_ratio(q_clean, w_clean)
        if ts >= 70:
            boost(w, ts)
        if pr >= 70:
            boost(w, pr)

    for scorer in [fuzz.token_sort_ratio, fuzz.partial_ratio]:
        for w, sc, _ in process.extract(q, vocab, scorer=scorer, limit=200):
            if sc >= 60:
                boost(w, sc)

    # STEP 4: Levenshtein distance
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                boost(w, 90 - 10 * d)

    # STEP 5: filter out irrelevant short words
    final = [(w, s) for w, s in scores.items()
             if len(w) > 2 or s >= 90]

    ranked = sorted(final, key=lambda kv: -kv[1])
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
