from fuzzywuzzy import process
import string
from flask import Flask, render_template, request
import os

application = Flask(__name__)
application.config["TEMPLATES_AUTO_RELOAD"] = True
application.config["DEBUG"] = True

# Load and process Finnegans Wake once
TEXT_PATH = "finneganswake.txt"
if not os.path.exists(TEXT_PATH):
    raise FileNotFoundError(f"{TEXT_PATH} not found. Make sure it exists in the working directory.")

with open(TEXT_PATH, "r") as f:
    text = f.read()

text_lines = text.split('\n')
text_words = []
text_words_lower = []
word_mapping = {}

for pos, line in enumerate(text_lines):
    for word in line.split():
        text_words.append(word)
        cleaned = word.translate(str.maketrans('', '', string.punctuation))
        if cleaned:
            lw = cleaned.lower()
            text_words_lower.append(lw)
            word_mapping.setdefault(lw, []).append(dict(line=pos, twlp=len(text_words) - 1))

@application.route("/", methods=["GET", "POST"])
def index():
    if request.method == "HEAD":
        return "\n"
    if request.method == "GET":
        return render_template("index.html")

    search_word = request.form.get("searchWord", "").strip()
    match_positions = []

    if search_word:
        matches = process.extractBests(search_word.lower(), text_words_lower, score_cutoff=65, limit=3000)
        seen = set()
        for match, score in matches:
            if match not in seen:
                seen.add(match)
                positions = word_mapping.get(match, [])
                mapped = [dict(word=text_words[p["twlp"]], positions=p["line"]) for p in positions]
                match_positions.append(dict(match=match, positions=mapped))

    return render_template("index.html", match=match_positions, search_word=search_word)

@application.route("/finneganswake", methods=["GET"])
def finneganswake():
    return render_template("finneganswake.html")

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=80)
