import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
import jellyfish
from doublemetaphone import doublemetaphone

# --- Configuration constants -------------------------
High_Freq_Cutoff    = 6     # any 1–3 letter word occurring ≥ this will be demoted
DEFAULT_MAX_RESULTS = 700   # cap on total results

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
        w = stripped.lower()
        vocab.add(w)
        positions.setdefault(w, []).append({
            "word": raw,
            "line": lineno
        })

# Build phonetic buckets via Double Metaphone
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


def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    # 1) substring boost (raw & cleaned)
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        if q in w or q_clean in w_clean:
            scores[w] = max(scores.get(w, 0), 100)

    # 2) fuzzy token_sort_ratio (≥50)
    for w, score, _ in process.extract(q, vocab,
                                       scorer=fuzz.token_sort_ratio,
                                       limit=200):
        if score >= 50:
            scores[w] = max(scores.get(w, 0), score)

    # 3) fuzzy partial_ratio (≥50)
    for w, score, _ in process.extract(q, vocab,
                                       scorer=fuzz.partial_ratio,
                                       limit=200):
        if score >= 50:
            scores[w] = max(scores.get(w, 0), score)

    # 4) token_set_ratio on cleaned strings (≥60)
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        score = fuzz.token_set_ratio(q_clean, w_clean)
        if score >= 60:
            scores[w] = max(scores.get(w, 0), score)

    # 5) Levenshtein distance
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                lev_score = 100 - (d * 10)
                scores[w] = max(scores.get(w, 0), lev_score)

    # 6) bigram overlap on raw strings (≥0.5)
    for w in vocab:
        ov = ngram_overlap(q, w)
        if ov >= 0.5:
            scores[w] = max(scores.get(w, 0), int(ov * 100))

    # 7) Jaro–Winkler on cleaned strings (≥0.80)
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        jw = jellyfish.jaro_winkler_similarity(q_clean, w_clean)
        if jw >= 0.80:
            scores[w] = max(scores.get(w, 0), int(jw * 100))

    # 8) exact phonetic match → boost to 100
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                scores[w] = max(scores.get(w, 0), 100)

    # 9) rank by score descending
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])

    # 10) demote: all 1–2 letter words, and any 3-letter word ≥ cutoff
    tail_set = {
        w
        for w, _ in ranked
        if len(w) <= 2
           or (len(w) == 3 and len(positions.get(w, [])) >= High_Freq_Cutoff)
    }

    # 11) primary list: everything except those in tail_set
    primary = [w for w, _ in ranked if w not in tail_set]

    # 12) tail list: only those demoted
    tail = [w for w, _ in ranked if w in tail_set]

    # 13) final ordering + cap
    ordered = primary + tail
    return ordered[:max_results]


# --- Flask routes -----------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_word = request.form.get("searchWord", "").strip()
        if not search_word:
            return render_template("index.html", match=[], search_word="")
        matches = find_matches(search_word, vocab, phonetic_buckets)
        out = [{"match": w, "positions": positions[w]} for w in matches]
        return render_template("index.html", match=out, search_word=search_word)
    return render_template("index.html", match=[], search_word="")


@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True) or {}
    q = data.get("query", "").strip()
    if not q:
        return jsonify([])
    matches = find_matches(q, vocab, phonetic_buckets)
    return jsonify([{"match": w, "positions": positions[w]} for w in matches])


@app.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html", lines=enumerate(lines))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
