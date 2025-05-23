import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import fuzz
from rapidfuzz import process
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration constants -------------------------
High_Freq_Cutoff    = 6     # 3-letter words with ≥ this freq get demoted
DEFAULT_MAX_RESULTS = 700   # maximum number of results to return

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # turn off in prod

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

# --- Utility: character n-gram overlap ---------------------
def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

# --- Core matching function -------------------------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    # cleaned form, no punctuation
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    # STEP 1: cleaned-prefix boost → 100
    prefix_len = min(4, len(q_clean))
    prefix = q_clean[:prefix_len]
    if prefix:
        for w in vocab:
            w_clean = re.sub(r"[^a-z0-9]", "", w)
            if w_clean.startswith(prefix):
                scores[w] = 100

    # STEP 2: cleaned-substring boost →  ninety-five
    if q_clean:
        for w in vocab:
            w_clean = re.sub(r"[^a-z0-9]", "", w)
            if q_clean in w_clean:
                scores[w] = max(scores.get(w, 0), 95)

    # STEP 3: cleaned fuzzy token_sort_ratio ≥ 70 → keep that score
    if q_clean:
        for w in vocab:
            w_clean = re.sub(r"[^a-z0-9]", "", w)
            ts = fuzz.token_sort_ratio(q_clean, w_clean)
            if ts >= 70:
                scores[w] = max(scores.get(w, 0), ts)

    # STEP 4: cleaned fuzzy partial_ratio ≥ 70
    if q_clean:
        for w in vocab:
            w_clean = re.sub(r"[^a-z0-9]", "", w)
            pr = fuzz.partial_ratio(q_clean, w_clean)
            if pr >= 70:
                scores[w] = max(scores.get(w, 0), pr)

    # STEP 5: raw fuzzy token_sort_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        if sc >= 60:
            scores[w] = max(scores.get(w, 0), sc)

    # STEP 6: raw fuzzy partial_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        if sc >= 60:
            scores[w] = max(scores.get(w, 0), sc)

    # STEP 7: bigram-overlap (raw) ≥ 0.5
    for w in vocab:
        ov = ngram_overlap(q, w)
        if ov >= 0.5:
            scores[w] = max(scores.get(w, 0), int(ov * 100))

    # STEP 8: Levenshtein distance (strict edits)
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                lev = 100 - (10 * d)
                scores[w] = max(scores.get(w, 0), lev)

    # STEP 9: exact Double-Metaphone → boost to 100
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                scores[w] = 100

    # FINAL SORT: by score desc, then length desc
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], -len(kv[0]))
    )

    # DEMOTION: only 1–2 letter words, and any 3-letter word over-frequent
    tail_set = {
        w for w, _ in ranked
        if len(w) <= 2 or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail_set]
    tail    = [w for w, _ in ranked if w in tail_set]

    return (primary + tail)[:max_results]

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
