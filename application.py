import difflib
import string
from flask import Flask, flash, redirect, render_template, request
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError

application = Flask(__name__)

# Ensure templates are auto-reloaded
application.config["TEMPLATES_AUTO_RELOAD"] = True
application.config["DEBUG"] = True

# route for main html page
@application.route("/", methods = ['GET', 'POST'])
def index():
    if request.method == 'HEAD':
        return "\n"
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        search_word = request.form.get("searchWord")

        # assigns list for function
        with open("finneganswake.txt", "r") as f:
            text = f.read()

        # splits file for function same as way html read text
        text_lines = text.split('\n')

        # word mapping to dictionary to prepare for hyperlink, stripped of punctuation
        text_words = []
        text_words_lower = []
        word_mapping = {}

        # iterates through lines and words on lines; twlp = position for every stripped word
        for pos, line in enumerate(text_lines):
            for word in line.split(' '):
                text_words.append(word)
                strip_word = word.translate(str.maketrans('', '', string.punctuation))
                if strip_word:
                    text_words_lower.append(strip_word.lower())
                    word_mapping[strip_word.lower()] = word_mapping.get(strip_word.lower(), []) + [dict(line = pos, twlp = len(text_words) -1)]


        # difflib function from python libraries for getting close matches
        n = 3000
        cutoff = 0.35
        matches = []
        if search_word:
            matches = difflib.get_close_matches(search_word.lower(), text_words_lower, n, cutoff)

        # gets position of every word, via each line, including any other characters, in matches to send to html rendering
        match_positions = []
        for match in set(matches):
            word_positions = word_mapping.get(match.lower())
            word_positions_mapping = []
            for position in word_positions:
                word_orig = text_words[position.get('twlp')]
                word_positions_mapping.append(dict(word = word_orig, positions = position.get('line')))
            match_positions.append(dict(match = match.lower(), positions = word_positions_mapping))
        return render_template('index.html', match = match_positions, search_word = search_word)

# new route for finnegans wake hyperlinked text
@application.route("/finneganswake", methods = ['GET'])
def finneganswake():
    return render_template('finneganswake.html')

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=80)
