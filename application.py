import re
import string

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
import Levenshtein
from doublemetaphone import doublemetaphone

# --- Configuration constants -------------------------
High_Freq_Cutoff    = 6     # any 1–3 letter word occurring ≥ this will be demoted
DEFAULT_MAX_RESULTS = 700   # cap on number of results

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False  # turn off in prod

# --- Pre-load text & build indexes -------------------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}  # word_lower → list of { line: int, word: original }
word_re = re.compile(r"\b[\w'-]+\b")
for lineno, line in enumerate(lines):
    for raw in word_re.findall(line):
        w = raw.strip(string.punctuation)
        if not w:
            continue
        wl = w.lower()
        vocab.add(wl)
        positions.setdefault(wl, []).append({"word": raw, "line": lineno})

# phonetic buckets via Double Metaphone
phonetic_buckets = {}
for w in vocab:
    p, s = doublemetaphone(w)
    for code in (p, s):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- ngram helper -----------------------------------------
def ngram_overlap(a: str, b: str, n: int = 2) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

# --- Matching pipeline ------------------------------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q = query.lower()
    # strip out any non-alphanumeric so hyphens/apostrophes vanish
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    scores = {}

    # STEP 1: cleaned-substring boost → 100
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        if q_clean and q_clean in w_clean:
            scores[w] = 100

    # STEP 2: cleaned-fuzzy partial_ratio ≥ 80 → 90
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        pr = fuzz.partial_ratio(q_clean, w_clean)
        if pr >= 80:
            scores[w] = max(scores.get(w, 0), 90)

    # STEP 3: cleaned-fuzzy token_set_ratio ≥ 80 → 85
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        ts = fuzz.token_set_ratio(q_clean, w_clean)
        if ts >= 80:
            scores[w] = max(scores.get(w, 0), 85)

    # STEP 4: cleaned-trigram-overlap ≥ 0.5 → 80
    for w in vocab:
        w_clean = re.sub(r"[^a-z0-9]", "", w)
        if ngram_overlap(q_clean, w_clean, n=3) >= 0.5:
            scores[w] = max(scores.get(w, 0), 80)

    # STEP 5: raw fuzzy token_sort_ratio ≥ 50
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_sort_ratio,
                                     limit=200):
        if sc >= 50:
            scores[w] = max(scores.get(w, 0), sc)

    # STEP 6: raw fuzzy partial_ratio ≥ 50
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.partial_ratio,
                                     limit=200):
        if sc >= 50:
            scores[w] = max(scores.get(w, 0), sc)

    # STEP 7: Levenshtein distance (small words) → 100-(d*10)
    L_THRESH = 2 if len(q) <= 5 else 3
    for w in vocab:
        if abs(len(w) - len(q)) <= L_THRESH:
            d = Levenshtein.distance(q, w)
            if d <= L_THRESH:
                lev = 100 - d*10
                scores[w] = max(scores.get(w, 0), lev)

    # STEP 8: raw bigram overlap ≥ 0.5 → int(ov*100)
    for w in vocab:
        ov = ngram_overlap(q, w)
        if ov >= 0.5:
            scores[w] = max(scores.get(w, 0), int(ov*100))

    # STEP 9: raw token_set_ratio ≥ 60
    for w, sc, _ in process.extract(q, vocab,
                                     scorer=fuzz.token_set_ratio,
                                     limit=200):
        if sc >= 60:
            scores[w] = max(scores.get(w, 0), sc)

    # STEP 10: exact Double-Metaphone → 100
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        if code:
            for w in phonetic_buckets.get(code, []):
                scores[w] = 100

    # FINAL SORT: by score desc, then length desc (longer first)
    ranked = sorted(scores.items(),
                    key=lambda kv: (-kv[1], -len(kv[0])))

    # DEMOTION: only 1–2 letter words and over-common 3-letter words
    tail = {
        w for w, _ in ranked
        if len(w) <= 2
           or (len(w) == 3 and len(positions[w]) >= High_Freq_Cutoff)
    }

    primary = [w for w, _ in ranked if w not in tail]
    tail_list = [w for w, _ in ranked if w in tail]

    return (primary + tail_list)[:max_results]


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
