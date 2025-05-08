import string
import re
from flask import Flask, render_template, request, jsonify
from fuzzywuzzy import fuzz, process

application = Flask(__name__)

# Ensure templates are auto-reloaded
application.config["TEMPLATES_AUTO_RELOAD"] = True
application.config["DEBUG"] = False  # turn off debug in production

def is_valid_query(query):
    # Accept only letters, numbers, and common punctuation, max 100 characters
    return bool(re.match(r"^[\w\s\.,'-]{1,100}$", query))

def search_finnegans_wake(search_word):
    # Load text
    with open("finneganswake.txt", "r") as f:
        text = f.read()

    # Split lines and words
    text_lines  = text.split("\n")
    text_words  = []
    word_mapping = {}

    for line_no, line in enumerate(text_lines):
        for raw in line.split(" "):
            text_words.append(raw)
            key = raw.translate(str.maketrans("", "", string.punctuation)).lower()
            if key:
                word_mapping.setdefault(key, []).append({
                    "word": raw,
                    "line": line_no
                })

    # Use fuzzywuzzy to get close matches
    # process.extract returns (candidate, score) tuples
    # We only keep those above a threshold (e.g. 35%)
    all_keys = list(word_mapping.keys())
    limit = 3000
    threshold = 35  # percent

    matches = [
        match for match, score in process.extract(
            search_word.lower(),
            all_keys,
            limit=limit,
            scorer=fuzz.WRatio
        )
        if score >= threshold
    ]

    # Build positions for unique matches
    results = []
    for key in set(matches):
        entries = word_mapping.get(key, [])
        positions = [
            {"word": e["word"], "positions": e["line"]}
            for e in entries
        ]
        results.append({"match": key, "positions": positions})

    return results

# Main page
@application.route("/", methods=["GET", "POST", "HEAD"])
def index():
    if request.method == "HEAD":
        return "\n"
    if request.method == "GET":
        return render_template("index.html")
    # POST
    search_word = request.form.get("searchWord", "").strip()
    if not is_valid_query(search_word):
        # You could flash an error or render with an error message
        return render_template("index.html", error="Invalid input", search_word=search_word)

    match_positions = search_finnegans_wake(search_word)
    return render_template("index.html", match=match_positions, search_word=search_word)

# JSON API endpoint
@application.route("/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()

    if not is_valid_query(query):
        return jsonify({"error": "Invalid input"}), 400

    results = search_finnegans_wake(query)
    return jsonify(results), 200

# Static hyperlinked text
@application.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html")

if __name__ == "__main__":
    # Development server (Gunicorn should be used in production)
    application.run(host="0.0.0.0", port=5000)
