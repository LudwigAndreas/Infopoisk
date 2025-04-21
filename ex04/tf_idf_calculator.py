import os
import math
import collections
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

def read_tokens(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def compute_tf(tokens):
    total_terms = len(tokens)
    counts = collections.Counter(tokens)
    return {term: count / total_terms for term, count in counts.items()}

def compute_idf(documents):
    """documents: list of sets of unique terms or lemmas from each doc"""
    N = len(documents)
    idf = {}
    all_terms = set(term for doc in documents for term in doc)
    for term in all_terms:
        containing_docs = sum(1 for doc in documents if term in doc)
        idf[term] = math.log(N / (1 + containing_docs)) + 1
    return idf

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_map = {}  # {number: tokens}
    for fname in os.listdir(input_dir):
        if fname.startswith("tokens_") and fname.endswith(".txt"):
            number = fname.replace("tokens_", "").replace(".txt", "")
            tokens = read_tokens(os.path.join(input_dir, fname))
            file_map[number] = tokens

    # TF for terms and lemmas
    term_tfs = {}
    lemma_tfs = {}
    term_sets = []
    lemma_sets = []

    for num, tokens in file_map.items():
        term_tf = compute_tf(tokens)
        lemmas = lemmatize_tokens(tokens)
        lemma_tf = compute_tf(lemmas)

        term_tfs[num] = term_tf
        lemma_tfs[num] = lemma_tf

        term_sets.append(set(term_tf.keys()))
        lemma_sets.append(set(lemma_tf.keys()))

    # IDF across all documents
    term_idf = compute_idf(term_sets)
    lemma_idf = compute_idf(lemma_sets)

    # Save TF-IDF files
    for num in file_map:
        with open(os.path.join(output_dir, f"tfidf_terms_{num}.txt"), "w", encoding="utf-8") as f:
            for term, tf in term_tfs[num].items():
                idf = term_idf.get(term, 0)
                tfidf = tf * idf
                f.write(f"{term} {idf:.6f} {tfidf:.6f}\n")

        with open(os.path.join(output_dir, f"tfidf_lemmas_{num}.txt"), "w", encoding="utf-8") as f:
            for lemma, tf in lemma_tfs[num].items():
                idf = lemma_idf.get(lemma, 0)
                tfidf = tf * idf
                f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate TF-IDF for terms and lemmas.")
    parser.add_argument("input_dir", help="Directory with tokens_{number}.txt files")
    parser.add_argument("output_dir", help="Directory to save tf-idf result files")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)
