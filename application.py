import re
from flask import Flask, request, jsonify
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

app = Flask(__name__)

# Load Finnegans Wake once at startup
with open("finneganswake.txt", encoding="utf-8") as f:
    finnegans_lines = [line.strip() for line in f if line.strip()]

def is_valid_query(query):
    # Accept only letters, numbers, and common punctuation, max 100 characters
    return bool(re.match(r"^[\w\s.,'-]{1,100}$", query))

def search_finnegans_wake(query):
    # Use fuzzywuzzy to find best matches (returns list of (match, score))
    matches = process.extract(query, finnegans_lines, scorer=fuzz.token_sort_ratio, limit=10)
    # Format results
    return [{"line": match[0], "score": match[1]} for match in matches if match[1] > 50]

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")

    if not is_valid_query(query):
        return jsonify({"error": "Invalid input"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 
