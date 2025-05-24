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
High_Freq_Cutoff    = 6      
DEFAULT_MAX_RESULTS = 700    

# length‐bonus scale
LENGTH_BONUS_MAX   = 50     
LENGTH_BONUS_STEP  = 10     

β_LENGTH           = 0.1    

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
        stripped = raw.strip(string.punctuation)
        if not stripped:
            continue
        w = stripped.lower()
        vocab.add(w)
        positions.setdefault(w, []).append({"word": raw, "line": lineno})

# --- Build Double Metaphone buckets ----------
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

freq     = {w: len(positions[w]) for w in vocab}
max_freq = max(freq.values())

def length_factor(L: int, Q: int) -> float:
    if L <= Q:
        return L / Q
    return 1 + β_LENGTH * ((L / Q) - 1)

def freq_factor(w: str) -> float:
    f = freq[w] / max_freq
    return max(0.0, 1 - math.sqrt(f))

# --- Core matching pipeline --------------
def find_matches(query, vocab, phonetic_buckets,
                 max_results=DEFAULT_MAX_RESULTS):
    q       = query.lower().strip()
    q_clean = re.sub(r"[^a-z0-9]", "", q)
    if not q_clean:
        return []

    # truncate at apostrophe
    cleaned = {}
    for w in vocab:
        base = w.split("'", 1)[0]
        base_clean = re.sub(r"[^a-z0-9]", "", base)
        if base_clean:
            cleaned[w] = base_clean

    candidates = list(cleaned.keys())
    raw_scores = {}

    def boost(w, sc):
        raw_scores[w] = max(raw_scores.get(w, 0), sc)

    # A) substring → 100
    for w, w_cl in cleaned.items():
        if q_clean in w_cl:
            boost(w, 100)

    # B) prefix → 110
    for w, w_cl in cleaned.items():
        if w_cl.startswith(q_clean):
            boost(w, 110)

    # C) partial‐prefix (3) → 105
    pre3 = q_clean[:3]
    for w, w_cl in cleaned.items():
        if w_cl.startswith(pre3) and not w_cl.startswith(q_clean):
            boost(w, 105)

    # D) first‐4 substring → 100
    first4 = q_clean[:4]
    for w, w_cl in cleaned.items():
        if (first4 in w_cl
            and not w_cl.startswith(q_clean)
            and not w_cl.startswith(pre3)):
            boost(w, 100)

    # E) suffix match → 105  ← *this is now unconditional*
    SUB_LEN = max(4, len(q_clean)//2)
    suffix  = q_clean[-SUB_LEN:]
    for w, w_cl in cleaned.items():
        if w_cl.endswith(suffix):
            boost(w, 105)

    # F) transpositions & fuzzy/phonetic as before…
    for w, w_cl in cleaned.items():
        if DamerauLevenshtein.distance(q_clean, w_cl) == 1:
            boost(w, 105)
        elif DamerauLevenshtein.distance(q_clean, w_cl) == 2:
            boost(w, 90)

    for w, w_cl in cleaned.items():
        jw = jellyfish.jaro_winkler_similarity(q_clean, w_cl)
        if jw >= 0.80:
            boost(w, int(jw*100))

    for w, w_cl in cleaned.items():
        ov = ngram_overlap(q_clean, w_cl)
        if ov >= 0.6:
            boost(w, int(ov*100))
        ts = fuzz.token_set_ratio(q_clean, w_cl)
        if ts >= 70:
            boost(w, ts)
        pr = fuzz.partial_ratio(q_clean, w_cl)
        if pr >= 70:
            boost(w, pr)

    for scorer in (fuzz.token_sort_ratio, fuzz.partial_ratio):
        for w, sc, _ in process.extract(q, candidates, scorer=scorer, limit=200):
            if sc >= 60:
                boost(w, sc)

    thresh = 2 if len(q_clean) <=5 else 3
    for w in candidates:
        w_cl = cleaned[w]
        if abs(len(w_cl)-len(q_clean)) <= thresh:
            d = Levenshtein.distance(q_clean, w_cl)
            if d <= thresh:
                boost(w, 100 - d*10)

    pcode, scode = doublemetaphone(q_clean)
    for code in (pcode, scode):
        for w in phonetic_buckets.get(code, []):
            if w in cleaned:
                boost(w, 95)

    # O) length bonus (50/10)
    Qlen = len(q_clean)
    for w, w_cl in cleaned.items():
        if w in raw_scores:
            diff  = abs(len(w_cl)-Qlen)
            bonus = max(0, LENGTH_BONUS_MAX - diff*LENGTH_BONUS_STEP)
            raw_scores[w] += bonus

    # normalize, sort, tail‐demote & exact‐bypass exactly as before…

    normalized = []
    for w, raw in raw_scores.items():
        L  = len(cleaned[w])
        normalized.append((w, raw * length_factor(L,Qlen) * freq_factor(w)))

    normalized.sort(key=lambda x:(-x[1], -len(x[0])))

    tail   = {w for w,_ in normalized
              if len(w)<=2 or (len(w)==3 and freq[w]>=High_Freq_Cutoff)}
    primary   = [w for w,_ in normalized if w not in tail]
    tail_list = [w for w,_ in normalized if w in tail]
    ordered   = primary + tail_list

    exacts = [w for w,w_cl in cleaned.items() if w_cl==q_clean]
    exacts.sort(key=lambda w:min(p["line"] for p in positions[w]))

    final = []
    for w in exacts + ordered:
        if w not in final:
            final.append(w)

    if len(q_clean)>1:
        final = [w for w in final
                 if len(cleaned[w])>1 and w.isascii()]

    return final[:max_results]

# --- Flask routes ---------------------------
@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        q = request.form.get("searchWord","").strip()
        if not q:
            return render_template("index.html", match=[], search_word="")
        ms  = find_matches(q, vocab, phonetic_buckets)
        out = [{"match":w,"positions":positions[w]} for w in ms]
        return render_template("index.html",match=out,search_word=q)
    return render_template("index.html",match=[],search_word="")

@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json(force=True) or {}
    q = data.get("query","").strip()
    if not q:
        return jsonify([])
    ms = find_matches(q, vocab, phonetic_buckets)
    return jsonify([{"match":w,"positions":positions[w]} for w in ms])

@app.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html", lines=enumerate(lines))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)
