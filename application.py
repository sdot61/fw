import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration -------------------------
High_Freq_Cutoff    = 6     # demote 3-letter words occurring ≥ this
DEFAULT_MAX_RESULTS = 700   # how many to return max

# --- Flask setup ---------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False

# --- Load & index the Wake -----------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}
word_re = re.compile(r"\b[\w'-]+\b")
for lineno, line in enumerate(lines):
    for raw in word_re.findall(line):
        stripped = raw.strip(string.punctuation)
        if not stripped:
            continue
        w = stripped.lower()
        vocab.add(w)
        positions.setdefault(w, []).append({"word": raw, "line": lineno})

# --- Build Double-Metaphone buckets -------
phonetic_buckets = {}
for w in vocab:
    p, s = doublemetaphone(w)
    for code in (p, s):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- n-gram overlap helper ----------------
def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    grams_a = {a[i:i+n] for i in range(len(a)-n+1)}
    grams_b = {b[i:i+n] for i in range(len(b)-n+1)}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / min(len(grams_a), len(grams_b))

# --- The new find_matches -----------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower().strip()
    # cleaned version: drop punctuation entirely
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    def boost(w, score):
        scores[w] = max(scores.get(w, 0), score)

    # precompute cleaned forms
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}

    # STEP 1: cleaned-prefix boost → 100
    if q_clean:
        # use half the query length (min 3 chars) as prefix length
        prefix_n = max(3, len(q_clean) // 2)
        prefix_n = min(prefix_n, len(q_clean))
        prefix = q_clean[:prefix_n]
        for w, w_cl in cleaned.items():
            if w_cl.startswith(prefix):
                boost(w, 100)

    # STEP 2: cleaned-substring boost → 95
    if q_clean:
        for w, w_cl in cleaned.items():
            if q_clean in w_cl:
                boost(w, 95)

    # STEP 3: cleaned-fuzzy token_set_ratio ≥ 70 → up to 100
    if q_clean:
        for w, w_cl in cleaned.items():
            ts = fuzz.token_set_ratio(q_clean, w_cl)
            if ts >= 70:
                boost(w, ts)

    # STEP 4: cleaned-fuzzy partial_ratio ≥ 70 → up to 100
    if q_clean:
        for w, w_cl in cleaned.items():
            pr = fuzz.partial_ratio(q_clean, w_cl)
            if pr >= 70:
                boost(w, pr)

    # STEP 5: cleaned-trigram overlap ≥ 0.6 → up to 100
    if q_clean:
        for w, w_cl in cleaned.items():
            ov = ngram_overlap(q_clean, w_cl, n=3)
            if ov >= 0.6:
                boost(w, int(ov * 100))

    # STEP 6: raw fuzzy token_sort_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # STEP 7: raw fuzzy partial_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # STEP 8: Levenshtein distance (≤2 edits short, ≤3 long)
    L = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L:
            d = Levenshtein.distance(q, w)
            if d <= L:
                boost(w, 100 - (d * 10))

    # STEP 9: exact Double-Metaphone boost → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # FINAL SORT: by score desc, then by word length desc
    ranked = sorted(scores.items(),
                    key=lambda kv: (-kv[1], -len(kv[0])))

    # DEMOTE only 1–2 letters and over-common 3-letters       
    tail = {
        w for w, _ in ranked
        if len(w) <= 2
           or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail]
    tail_list = [w for w, _ in ranked if w in tail]

    return (primary + tail_list)[:max_results]


# --- Flask routes ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        q = request.form.get("searchWord", "").strip()
        if not q:
            return render_template("index.html", match=[], search_word="")
        m = find_matches(q, vocab, phonetic_buckets)
        out = [{"match": w, "positions": positions[w]} for w in m]
        return render_template("index.html", match=out, search_word=q)
    return render_template("index.html", match=[], search_word="")

@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True) or {}
    q = data.get("query", "").strip()
    if not q:
        return jsonify([])
    m = find_matches(q, vocab, phonetic_buckets)
    return jsonify([{"match": w, "positions": positions[w]} for w in m])

@app.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html", lines=enumerate(lines))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
