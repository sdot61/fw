import re
import string
from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz, process
import difflib

app = Flask(__name__)

# Read the text for search
with open("finneganswake.txt", "r") as f:
    text = f.read()

# Split text into lines for context retrieval
text_lines = text.split('\n')

# Word mapping to dictionary to prepare for hyperlinking
text_words = []
text_words_lower = []
word_mapping = {}

# Process text for word position mapping (word -> line number)
for pos, line in enumerate(text_lines):
    for word in line.split(' '):
        text_words.append(word)
        strip_word = word.translate(str.maketrans('', '', string.punctuation))
        if strip_word:
            text_words_lower.append(strip_word.lower())
            word_mapping[strip_word.lower()] = word_mapping.get(strip_word.lower(), []) + [
                dict(line=pos, twlp=len(text_words) - 1)]

def is_valid_query(query):
    """Accept only letters, numbers, and common punctuation, max 100 characters"""
    return bool(re.match(r"^[\w\s.,'-]{1,100}$", query))

def search_finnegans_wake(query):
    """Search for words or phrases in Finnegans Wake, return context"""
    n = 3000
    cutoff = 0.35
    matches = []
    match_positions = []

    # If query is a phrase, search for the exact phrase first
    if ' ' in query:
        for line in text_lines:
            if query.lower() in line.lower():
                match_positions.append({'line': line})
        return match_positions

    # If query is a single word, use fuzzy matching
    if query:
        matches = process.extract(query.lower(), text_words_lower, limit=n, scorer=fuzz.partial_ratio)
    
    # For each match, find positions in the text and add context
    for match, score in matches:
        if score >= cutoff:
            word_positions = word_mapping.get(match)
            word_positions_mapping = []
            for position in word_positions:
                word_orig = text_words[position.get('twlp')]
                line = text_lines[position.get('line')]
                word_positions_mapping.append(dict(
                    word=word_orig,
                    positions=position.get('line'),
                    context=line
                ))
            match_positions.append(dict(match=match, positions=word_positions_mapping))

    return match_positions

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')

    if not is_valid_query(query):
        return jsonify({"error": "Invalid input"}), 400

    # Run safe search logic
    results = search_finnegans_wake(query)
    return jsonify(results)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        search_word = request.form.get("searchWord")
        results = search_finnegans_wake(search_word)

        # Render the results to the user with highlights
        return render_template('index.html', results=results, search_word=search_word)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
