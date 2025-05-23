import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration -------------------------
High_Freq_Cutoff    = 6     # demote 3-letter words occurring ≥ this
DEFAULT_MAX_RESULTS = 700   # cap on total results

# --- Flask setup ---------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # off in prod

# --- Load & index the text -----------------
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

# --- Build phonetic buckets via Double Metaphone -----------
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- n-gram overlap helper ----------------
def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    grams_a = {a[i:i+n] for i in range(len(a)-n+1)}
    grams_b = {b[i:i+n] for i in range(len(b)-n+1)}
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / min(len(grams_a), len(grams_b))

# --- Core matching pipeline ----------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower().strip()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    def boost(w, sc):
        scores[w] = max(scores.get(w, 0), sc)

    # precompute cleaned forms once
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}

    # STEP 1: cleaned-substring boost → 100
    if q_clean:
        for w, w_cl in cleaned.items():
            if q_clean in w_cl:
                boost(w, 100)

    # STEP 2: light prefix bump → +10 (only if already scored)
    if q_clean:
        for w, w_cl in cleaned.items():
            if scores.get(w, 0) > 0 and w_cl.startswith(q_clean):
                scores[w] += 10

    # STEP 3: cleaned fuzzy ratio ≥ 75
    if q_clean:
        for w, w_cl in cleaned.items():
            sc = fuzz.ratio(q_clean, w_cl)
            if sc >= 75:
                boost(w, sc)

    # STEP 4: cleaned token_set_ratio ≥ 70
    if q_clean:
        for w, w_cl in cleaned.items():
            sc = fuzz.token_set_ratio(q_clean, w_cl)
            if sc >= 70:
                boost(w, sc)

    # STEP 5: cleaned partial_ratio ≥ 70
    if q_clean:
        for w, w_cl in cleaned.items():
            sc = fuzz.partial_ratio(q_clean, w_cl)
            if sc >= 70:
                boost(w, sc)

    # STEP 6: cleaned trigram overlap ≥ 0.6
    if q_clean:
        for w, w_cl in cleaned.items():
            ov = ngram_overlap(q_clean, w_cl, n=3)
            if ov >= 0.6:
                boost(w, int(ov * 100))

    # STEP 7: raw fuzzy token_sort_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # STEP 8: raw fuzzy partial_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # STEP 9: Levenshtein distance (≤2 edits if short, ≤3 if long)
    thresh = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= thresh:
            d = Levenshtein.distance(q, w)
            if d <= thresh:
                boost(w, 100 - (d * 10))

    # STEP 10: exact Double-Metaphone → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # --- Final sort: by score desc, then by length desc ---
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], -len(kv[0]))
    )

    # --- Demotion: only 2-letter tokens and over-common 3-letters ---
    tail_set = {
        w for w, _ in ranked
        if len(w) == 2 or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail_set]
    tail    = [w for w, _ in ranked if w in tail_set]
    ordered = primary + tail

    # --- Filter out single-letter tokens (unless query itself is 1 letter) ---
    if len(q_clean) > 1:
        ordered = [w for w in ordered if len(w) > 1]

    return ordered[:max_results]

# --- Flask routes ---------------------------
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
