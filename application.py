import difflib
import string
import re
from flask import Flask, flash, redirect, render_template, request, jsonify, abort
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError

application = Flask(__name__)
application.config["TEMPLATES_AUTO_RELOAD"] = True
application.config["DEBUG"] = True


def is_valid_query(query):
    # Accept only letters, numbers, spaces, and safe punctuation, max 100 characters
    return bool(re.match(r"^[\w\s.,'-]{1,100}$", query))


@application.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'HEAD':
        return "\n"
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        search_word = request.form.get("searchWord", "")

        if not is_valid_query(search_word):
            return render_template("index.html", error="Invalid input.")

        with open("finneganswake.txt", "r") as f:
            text = f.read()

        text_lines = text.split('\n')
        text_words = []
        text_words_lower = []
        word_mapping = {}

        for pos, line in enumerate(text_lines):
            for word in line.split(' '):
                text_words.append(word)
                strip_word = word.translate(str.maketrans('', '', string.punctuation))
                if strip_word:
                    text_words_lower.append(strip_word.lower())
                    word_mapping[strip_word.lower()] = word_mapping.get(strip_word.lower(), []) + [
                        dict(line=pos, twlp=len(text_words) - 1)]

        n = 3000
        cutoff = 0.35
        matches = []
        if search_word:
            matches = difflib.get_close_matches(search_word.lower(), text_words_lower, n, cutoff)

        match_positions = []
        for match in set(matches):
            word_positions = word_mapping.get(match.lower())
            word_positions_mapping = []
            for position in word_positions:
                word_orig = text_words[position.get('twlp')]
                word_positions_mapping.append(dict(word=word_orig, positions=position.get('line')))
            match_positions.append(dict(match=match.lower(), positions=word_positions_mapping))
        return render_template('index.html', match=match_positions, search_word=search_word)


@application.route("/search", methods=['POST'])
def api_search():
    query = request.json.get("query", "")

    if not is_valid_query(query):
        return jsonify({"error": "Invalid input"}), 400

    # Same logic as above, just returns JSON
    with open("finneganswake.txt", "r") as f:
        text = f.read()
    text_lines = text.split('\n')
    text_words = []
    text_words_lower = []
    word_mapping = {}

    for pos, line in enumerate(text_lines):
        for word in line.split(' '):
            text_words.append(word)
            strip_word = word.translate(str.maketrans('', '', string.punctuation))
            if strip_word:
                text_words_lower.append(strip_word.lower())
                word_mapping[strip_word.lower()] = word_mapping.get(strip_word.lower(), []) + [
                    dict(line=pos, twlp=len(text_words) - 1)]

    n = 3000
    cutoff = 0.35
    matches = []
    if query:
        matches = difflib.get_close_matches(query.lower(), text_words_lower, n, cutoff)

    match_positions = []
    for match in set(matches):
        word_positions = word_mapping.get(match.lower())
        word_positions_mapping = []
        for position in word_positions:
            word_orig = text_words[position.get('twlp')]
            word_positions_mapping.append(dict(word=word_orig, positions=position.get('line')))
        match_positions.append(dict(match=match.lower(), positions=word_positions_mapping))
    return jsonify({"search_word": query, "results": match_positions})


@application.route("/finneganswake", methods=['GET'])
def finneganswake():
    return render_template('finneganswake.html')


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=80)
