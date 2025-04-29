import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def load_tfidf_vectors(directory: str, use_lemmas: bool = True) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Load TF-IDF vectors from files.
    Returns:
        - doc_vectors: Dictionary mapping document numbers to their term/lemma vectors
        - idf_weights: Dictionary mapping terms/lemmas to their IDF weights
    """
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

def compute_query_vector(query: str, idf_weights: Dict[str, float], use_lemmas: bool = True) -> Dict[str, float]:
    """Convert query into TF-IDF vector."""
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # Tokenize and optionally lemmatize query
    tokens = word_tokenize(query.lower())
    if use_lemmas:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
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

def search(query: str, doc_vectors: Dict[str, Dict[str, float]], 
          idf_weights: Dict[str, float], urls: Dict[str, str],
          top_k: int = 10, use_lemmas: bool = True) -> List[Tuple[str, str, float]]:
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

def main():
    parser = argparse.ArgumentParser(description="Search documents using TF-IDF vectors")
    parser.add_argument("--tfidf_dir", required=True, help="Directory containing TF-IDF files")
    parser.add_argument("--index_file", required=True, help="File containing document numbers and URLs")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--use_lemmas", action="store_true", help="Use lemmas instead of terms")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Load vectors and URLs
    doc_vectors, idf_weights = load_tfidf_vectors(args.tfidf_dir, args.use_lemmas)
    urls = load_urls(args.index_file)
    
    # Perform search
    results = search(args.query, doc_vectors, idf_weights, urls, args.top_k, args.use_lemmas)
    
    # Print results
    print(f"\nSearch results for query: '{args.query}'")
    print("-" * 80)
    for doc_num, url, score in results:
        print(f"Document {doc_num}: {url}")
        print(f"Relevance score: {score:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 