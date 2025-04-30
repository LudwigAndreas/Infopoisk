from flask import Flask, render_template, request, jsonify
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Global variables to store loaded data
doc_vectors = {}
idf_weights = {}
urls = {}

def load_tfidf_vectors(directory: str, use_lemmas: bool = True) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Load TF-IDF vectors from files."""
    doc_vectors = {}
    idf_weights = {}
    
    prefix = "tfidf_lemmas_" if use_lemmas else "tfidf_terms_"
    
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".txt"):
            doc_num = filename.replace(prefix, "").replace(".txt", "")
            doc_vectors[doc_num] = {}
            
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                for line in f:
                    term, idf, tfidf = line.strip().split()
                    doc_vectors[doc_num][term] = float(tfidf)
                    if term not in idf_weights:
                        idf_weights[term] = float(idf)
    
    return doc_vectors, idf_weights

def load_urls(index_file: str) -> Dict[str, str]:
    """Load document numbers to URLs mapping from index file."""
    urls = {}
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            num, url = line.strip().split()
            urls[num] = url
    return urls

def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def compute_query_vector(query: str, idf_weights: Dict[str, float], use_lemmas: bool = True) -> Dict[str, float]:
    """Convert query into TF-IDF vector."""
    # Tokenize and optionally lemmatize query
    tokens = word_tokenize(query.lower())
    if use_lemmas:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    
    # Compute TF
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    total_terms = len(tokens)
    tf = {term: count/total_terms for term, count in tf.items()}
    
    # Compute TF-IDF
    query_vector = {}
    for term, tf_val in tf.items():
        if term in idf_weights:
            query_vector[term] = tf_val * idf_weights[term]
    
    return query_vector

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1.keys()) | set(vec2.keys()))
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def search(query: str, top_k: int = 10, use_lemmas: bool = True) -> List[Tuple[str, str, float]]:
    """Search for documents most relevant to the query."""
    query_vector = compute_query_vector(query, idf_weights, use_lemmas)
    
    # Compute similarity scores
    scores = []
    for doc_num, doc_vector in doc_vectors.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:  # Only include documents with non-zero similarity
            scores.append((doc_num, urls.get(doc_num, "Unknown URL"), score))
    
    # Sort by score in descending order
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.form.get('query', '')
    use_lemmas = request.form.get('use_lemmas', 'true').lower() == 'true'
    top_k = int(request.form.get('top_k', 10))
    
    results = search(query, top_k, use_lemmas)
    
    # Format results for display
    formatted_results = []
    for doc_num, url, score in results:
        formatted_results.append({
            'url': url,
            'score': f"{score:.4f}",
            'doc_num': doc_num
        })
    
    return jsonify({
        'query': query,
        'results': formatted_results
    })

def init_app():
    """Initialize the application by loading necessary data."""
    global doc_vectors, idf_weights, urls
    
    # Load data
    tfidf_dir = "output"  # Directory containing TF-IDF files
    index_file = "index.txt"  # Path to index file
    
    doc_vectors, idf_weights = load_tfidf_vectors(tfidf_dir)
    urls = load_urls(index_file)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .search-container {
            text-align: center;
            margin-bottom: 20px;
        }
        #search-input {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        #search-button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4285f4;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #search-button:hover {
            background-color: #3367d6;
        }
        .results {
            margin-top: 20px;
        }
        .result-item {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .result-url {
            color: #1a0dab;
            text-decoration: none;
            font-size: 18px;
        }
        .result-url:hover {
            text-decoration: underline;
        }
        .result-score {
            color: #006621;
            font-size: 14px;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .options {
            margin: 10px 0;
        }
        .options label {
            margin-right: 15px;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <h1>Vector Search</h1>
        <form id="search-form">
            <input type="text" id="search-input" placeholder="Enter your search query...">
            <button type="submit" id="search-button">Search</button>
            <div class="options">
                <label>
                    <input type="checkbox" id="use-lemmas" checked>
                    Use Lemmas
                </label>
                <label>
                    Number of results:
                    <input type="number" id="top-k" value="10" min="1" max="50">
                </label>
            </div>
        </form>
    </div>
    
    <div class="loading" id="loading">
        Searching...
    </div>
    
    <div class="results" id="results">
        <!-- Results will be inserted here -->
    </div>

    <script>
        document.getElementById('search-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const query = document.getElementById('search-input').value;
            const useLemmas = document.getElementById('use-lemmas').checked;
            const topK = document.getElementById('top-k').value;
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            // Send search request
            fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    'query': query,
                    'use_lemmas': useLemmas,
                    'top_k': topK
                })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                
                // Display results
                const resultsDiv = document.getElementById('results');
                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<p>No results found.</p>';
                    return;
                }
                
                let html = '';
                data.results.forEach(result => {
                    html += `
                        <div class="result-item">
                            <a href="${result.url}" class="result-url" target="_blank">${result.url}</a>
                            <div class="result-score">Relevance score: ${result.score}</div>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').innerHTML = '<p>An error occurred during the search.</p>';
                console.error('Error:', error);
            });
        });
    </script>
</body>
</html>
        ''')
    
    # Initialize the application
    init_app()
    
    # Run the Flask app
    app.run(debug=True) 
