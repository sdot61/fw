import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
from rapidfuzz.distance import DamerauLevenshtein
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration -------------------------
High_Freq_Cutoff    = 6     # demote 3-letter words occurring ≥ this
DEFAULT_MAX_RESULTS = 700   # cap on total results

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

# --- Build phonetic buckets ---------------
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- n-gram overlap helper ----------------
def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

# --- Core matching pipeline ----------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower().strip()
    q_clean = re.sub(r"[^a-z0-9]", "", q)

    # precompute cleaned forms once
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}
    scores = {}

    def boost(w, sc):
        scores[w] = max(scores.get(w, 0), sc)

    # STEP 0: exact-clean match → 300
    if q_clean:
        for w, w_cl in cleaned.items():
            if w_cl == q_clean:
                boost(w, 300)

    # STEP 1: fuzzy-prefix bump → 110
    if q_clean:
        half = len(q_clean) // 2
        prefix_len = min(len(q_clean), max(4, half))
        prefix = q_clean[:prefix_len]
        for w, w_cl in cleaned.items():
            if len(w_cl) >= prefix_len:
                sc = fuzz.ratio(prefix, w_cl[:prefix_len])
                if sc >= 75:
                    boost(w, 110)

    # STEP 2: hard prefix boost → 100
    if q_clean:
        for w, w_cl in cleaned.items():
            if w_cl.startswith(q_clean):
                boost(w, 100)

    # STEP 3: full cleaned-substring → 95
    if q_clean:
        for w, w_cl in cleaned.items():
            if q_clean in w_cl:
                boost(w, 95)

    # STEP 4: cleaned global ratio ≥ 60
    if q_clean:
        for w, w_cl in cleaned.items():
            sc = fuzz.ratio(q_clean, w_cl)
            if sc >= 60:
                boost(w, sc)

    # STEP 5: Damerau–Levenshtein (swap=1 edit)
    if q_clean:
        for w, w_cl in cleaned.items():
            d = DamerauLevenshtein.distance(q_clean, w_cl)
            if d == 0:
                boost(w, 100)
            elif d == 1:
                boost(w, 95)
            elif d == 2:
                boost(w, 80)

    # STEP 6: cleaned token_set_ratio ≥ 70
    if q_clean:
        for w, w_cl in cleaned.items():
            sc = fuzz.token_set_ratio(q_clean, w_cl)
            if sc >= 70:
                boost(w, sc)

    # STEP 7: cleaned partial_ratio ≥ 70
    if q_clean:
        for w, w_cl in cleaned.items():
            sc = fuzz.partial_ratio(q_clean, w_cl)
            if sc >= 70:
                boost(w, sc)

    # STEP 8: cleaned trigram overlap ≥ 0.5
    if q_clean:
        for w, w_cl in cleaned.items():
            ov = ngram_overlap(q_clean, w_cl, n=3)
            if ov >= 0.5:
                boost(w, int(ov * 100))

    # STEP 9: raw fuzzy token_sort_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # STEP 10: raw fuzzy partial_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        if sc >= 60:
            boost(w, sc)

    # STEP 11: classical Levenshtein (≤2/≤3 edits)
    thresh = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= thresh:
            d = Levenshtein.distance(q, w)
            if d <= thresh:
                boost(w, 100 - d * 10)

    # STEP 12: exact phonetic match → 95
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                boost(w, 95)

    # --- Penalty for short words (length ≤ 4) ---
    # lighter penalty for 4-letter, heavier for shorter
    for w in list(scores):
        L = len(w)
        if q_clean and 1 < L <= 4 and cleaned[w] != q_clean:
            penalty = (5 - L) * 10  # L=4→5pt, L=3→10pt, L=2→15pt
            scores[w] = max(0, scores[w] - penalty)

    # --- Final sort: by (score desc, length desc) ---
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], -len(kv[0])))

    # --- Tail-demotion: only 2-letter or over-common 3-letter words ---
    tail_set = {
        w for w, _ in ranked
        if len(w) == 2 or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }
    primary = [w for w, _ in ranked if w not in tail_set]
    tail    = [w for w, _ in ranked if w in tail_set]
    ordered = primary + tail

    # --- Filter out single-letter & non-ASCII tokens ---
    if len(q_clean) > 1:
        ordered = [
            w for w in ordered
            if len(w) > 1 and w.isascii()
        ]

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
