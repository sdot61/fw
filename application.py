import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import fuzz, process
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration constants -------------------------
High_Freq_Cutoff    = 6     # any 3-letter word occurring ≥ this will be demoted
DEFAULT_MAX_RESULTS = 700   # cap on total results

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

# --- Build phonetic buckets via Double Metaphone ----------
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- Helper: character n-gram overlap ---------------------
def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

# --- Core matching function ------------------------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    # strip punctuation so "mata-harried" -> "mataharried"
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    def boost(w, score):
        scores[w] = max(scores.get(w, 0), score)

    # Precompute cleaned forms
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}

    # 1) cleaned-substring boost → 100
    if q_clean:
        for w, w_cl in cleaned.items():
            if q_clean in w_cl:
                boost(w, 100)

    # 2) phonetic boost → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # 3) cleaned fuzzy token_set_ratio ≥ 75 → keep that score
    if q_clean:
        for w, w_cl in cleaned.items():
            ts = fuzz.token_set_ratio(q_clean, w_cl)
            if ts >= 75:
                boost(w, ts)

    # 4) cleaned fuzzy partial_ratio ≥ 75
    if q_clean:
        for w, w_cl in cleaned.items():
            pr = fuzz.partial_ratio(q_clean, w_cl)
            if pr >= 75:
                boost(w, pr)

    # 5) raw fuzzy token_sort_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # 6) raw fuzzy partial_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # 7) raw token_set_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_set_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # 8) bigram overlap (raw) ≥ 0.5 → int(ov*100)
    for w in vocab:
        ov = ngram_overlap(q, w)
        if ov >= 0.5:
            boost(w, int(ov * 100))

    # 9) Levenshtein distance (≤2 edits short, ≤3 edits long)
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                boost(w, 100 - (d * 10))

    # --- sort by score desc, then length desc ---------------
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], -len(kv[0]))
    )

    # --- demote only 1–2 letters and over-common 3-letters ---
    tail_set = {
        w for w, _ in ranked
        if len(w) <= 2 or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }

    primary = [w for w, _ in ranked if w not in tail_set]
    tail    = [w for w, _ in ranked if w in tail_set]

    # --- final ordering + cap -------------------------------
    ordered = primary + tail
    return ordered[:max_results]

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
