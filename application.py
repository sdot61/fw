import re
import string
import math

from flask import Flask, request, jsonify, render_template
from rapidfuzz import process, fuzz
from rapidfuzz.distance import DamerauLevenshtein
import Levenshtein
import jellyfish
from doublemetaphone import doublemetaphone

# --- Configuration -------------------------
High_Freq_Cutoff    = 6     # demote very common 3-letter words
DEFAULT_MAX_RESULTS = 700   # cap on total results
β_LENGTH            = 0.2   # mild boost for longer words

# --- Flask setup ---------------------------
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["DEBUG"] = False

# --- Load & index Finnegans Wake ------------
with open("finneganswake.txt", "r") as f:
    lines = f.read().splitlines()

vocab = set()
positions = {}
word_re = re.compile(r"\b[\w'-]+\b")
for lineno, line in enumerate(lines):
    for raw in word_re.findall(line):
        stripped = raw.strip("".join(set(string.punctuation)))
        if not stripped:
            continue
        w = stripped.lower()
        vocab.add(w)
        positions.setdefault(w, []).append({"word": raw, "line": lineno})

# --- Build phonetic buckets via Double Metaphone ---
phonetic_buckets = {}
for w in vocab:
    pcode, scode = doublemetaphone(w)
    for code in (pcode, scode):
        if code:
            phonetic_buckets.setdefault(code, []).append(w)

# --- Helpers ---------------------------------
def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
    b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

# Precompute frequency map
freq     = {w: len(positions[w]) for w in vocab}
max_freq = max(freq.values())

def length_factor(L: int, Q: int) -> float:
    """Shorter than query: L/Q; longer: 1 + β_LENGTH*(L/Q - 1)."""
    if L <= Q:
        return L / Q
    return 1 + β_LENGTH * ((L / Q) - 1)

def freq_factor(w: str) -> float:
    """Down-weight very common tokens."""
    return max(0.0, 1 - (freq[w] / max_freq))

# --- Core matching pipeline --------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q       = query.lower().strip()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    if not q_clean:
        return []

    # cleaned forms
    cleaned = {w: re.sub(r"[^a-z0-9]", "", w) for w in vocab}

    # 1) Raw scoring
    raw_scores = {}
    def boost(w, sc):
        raw_scores[w] = max(raw_scores.get(w, 0), sc)

    # A) substring → 100
    for w, w_cl in cleaned.items():
        if q_clean in w_cl:
            boost(w, 100)

    # B) full-prefix → 120
    for w, w_cl in cleaned.items():
        if w_cl.startswith(q_clean):
            boost(w, 120)

    # C) partial-prefix (first 3 chars) → 115
    pre3 = q_clean[:3]
    for w, w_cl in cleaned.items():
        if w_cl.startswith(pre3) and not w_cl.startswith(q_clean):
            boost(w, 115)

    # D) first-4 substring → 110
    first4 = q_clean[:4]
    for w, w_cl in cleaned.items():
        if (first4 in w_cl
            and not w_cl.startswith(q_clean)
            and not w_cl.startswith(pre3)):
            boost(w, 110)

    # E) single-transposition → 105
    for w, w_cl in cleaned.items():
        if DamerauLevenshtein.distance(q_clean, w_cl) == 1:
            boost(w, 105)

    # F) Jaro–Winkler ≥ .80 → up to 100
    for w, w_cl in cleaned.items():
        jw = jellyfish.jaro_winkler_similarity(q_clean, w_cl)
        if jw >= 0.80:
            boost(w, int(jw * 100))

    # G) trigram-overlap ≥ .60 → up to 100
    for w, w_cl in cleaned.items():
        ov = ngram_overlap(q_clean, w_cl)
        if ov >= 0.6:
            boost(w, int(ov * 100))

    # H) fuzzy token_set_ratio ≥ 70 → up to 100
    for w, w_cl in cleaned.items():
        ts = fuzz.token_set_ratio(q_clean, w_cl)
        if ts >= 70:
            boost(w, ts)

    # I) fuzzy partial_ratio ≥ 70 → up to 100
    for w, w_cl in cleaned.items():
        pr = fuzz.partial_ratio(q_clean, w_cl)
        if pr >= 70:
            boost(w, pr)

    # J) raw fuzzy extracts (token_sort + partial) ≥ 60
    for scorer in (fuzz.token_sort_ratio, fuzz.partial_ratio):
        for w, sc, _ in process.extract(q, vocab, scorer=scorer, limit=200):
            if sc >= 60:
                boost(w, sc)

    # K) Levenshtein distance (≤2 short, ≤3 long) → up to 100
    thresh = 2 if len(q_clean) <= 5 else 3
    for w in vocab:
        w_cl = cleaned[w]
        if abs(len(w_cl) - len(q_clean)) <= thresh:
            d = Levenshtein.distance(q_clean, w_cl)
            if d <= thresh:
                boost(w, 100 - d * 10)

    # L) exact phonetic match → 120
    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        for w in phonetic_buckets.get(code, []):
            boost(w, 120)

    # 2) Normalize by length & frequency
    normalized = []
    Q = len(q_clean)
    for w, raw in raw_scores.items():
        L  = len(cleaned[w])
        lf = length_factor(L, Q)
        ff = freq_factor(w)
        normalized.append((w, raw * lf * ff))

    # 3) Sort by normalized score desc, then length desc
    normalized.sort(key=lambda x: (-x[1], -len(x[0])))

    # 4) Tail-demotion: 1–2 letters & over-common 3s
    tail = {
        w for w, _ in normalized
        if len(w) <= 2
           or (len(w) == 3 and freq[w] >= High_Freq_Cutoff)
    }
    primary   = [w for w, _ in normalized if w not in tail]
    tail_list = [w for w, _ in normalized if w in tail]
    ordered   = primary + tail_list

    # 5) exact-literal bypass at top
    exacts = [w for w, w_cl in cleaned.items() if w_cl == q_clean]
    exacts.sort(key=lambda w: min(p["line"] for p in positions[w]))
    final = []
    for w in exacts + ordered:
        if w not in final:
            final.append(w)

    # 6) filter single-letter & non-ASCII for multi-char queries
    if len(q_clean) > 1:
        final = [w for w in final if len(cleaned[w]) > 1 and w.isascii()]

    return final[:max_results]

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
