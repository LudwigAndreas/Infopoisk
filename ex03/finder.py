import os
import re
import sys
import json
import argparse
from collections import defaultdict

class SearchEngine:
    def __init__(self, directory_path):
        """Initialize the search engine with the specified directory."""
        self.directory_path = directory_path
        self.doc_urls = {}  # Maps document number to URL
        self.inverted_index = defaultdict(set)  # Maps terms to document numbers
        self.index_file = os.path.join(directory_path, "inverted_index.json")
        
        # Try to load existing index if it exists, otherwise build it
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            self.build_index()
            self.save_index()

    def build_index(self):
        """Build the inverted index from the files in the directory."""
        print("Building inverted index...")
        # First, read the index.txt file to get document numbers and URLs
        index_file_path = os.path.join(self.directory_path, "index.txt")
        try:
            with open(index_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        doc_number, url = parts
                        self.doc_urls[doc_number] = url
        except FileNotFoundError:
            print(f"Error: index.txt not found in {self.directory_path}")
            return

        # For each document, read tokens and add to the inverted index
        for doc_number in self.doc_urls:
            tokens_file = os.path.join(self.directory_path, f"tokens_{doc_number}.txt")
            lemmas_file = os.path.join(self.directory_path, f"lemmas_{doc_number}.txt")
            
            # Add tokens to the inverted index
            try:
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        token = line.strip()
                        if token:
                            self.inverted_index[token].add(doc_number)
            except FileNotFoundError:
                print(f"Warning: tokens file not found for document {doc_number}")
            
            # Add lemmas to the inverted index
            try:
                with open(lemmas_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            lemma = parts[0]
                            self.inverted_index[lemma].add(doc_number)
            except FileNotFoundError:
                print(f"Warning: lemmas file not found for document {doc_number}")
        
        print(f"Inverted index built with {len(self.inverted_index)} terms.")

    def save_index(self):
        """Save the inverted index to a file."""
        print(f"Saving inverted index to {self.index_file}...")
        # Convert sets to lists for JSON serialization
        serializable_index = {
            "doc_urls": self.doc_urls,
            "inverted_index": {term: list(docs) for term, docs in self.inverted_index.items()}
        }
        
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_index, f)
            print("Index saved successfully.")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self):
        """Load the inverted index from a file."""
        print(f"Loading inverted index from {self.index_file}...")
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.doc_urls = data["doc_urls"]
                # Convert lists back to sets
                self.inverted_index = {term: set(docs) for term, docs in data["inverted_index"].items()}
            print(f"Index loaded with {len(self.inverted_index)} terms.")
        except Exception as e:
            print(f"Error loading index: {e}")
            # If loading fails, build the index
            self.build_index()
            self.save_index()

    def _tokenize_query(self, query):
        """Tokenize a query into operators and terms."""
        # Replace parentheses with spaces around them for easier tokenization
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        # Tokenize the query
        tokens = query.split()
        return tokens

    def _evaluate_boolean_expression(self, tokens):
        """
        Evaluate a boolean expression using the Shunting Yard algorithm.
        Returns a set of document IDs that match the query.
        """
        output_queue = []
        operator_stack = []
        
        # Define operator precedence
        precedence = {'OR': 1, 'AND': 2, 'NOT': 3}
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()  # Discard the '('
            elif token in ['AND', 'OR', 'NOT']:
                while (operator_stack and 
                       operator_stack[-1] != '(' and 
                       operator_stack[-1] in precedence and 
                       precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            else:
                # It's a term, add it to the output queue
                output_queue.append(token)
            
            i += 1
        
        # Pop any remaining operators from the stack
        while operator_stack:
            output_queue.append(operator_stack.pop())
        
        # Evaluate the RPN expression
        eval_stack = []
        all_docs = set(self.doc_urls.keys())
        
        for token in output_queue:
            if token == 'AND':
                if len(eval_stack) >= 2:
                    right = eval_stack.pop()
                    left = eval_stack.pop()
                    eval_stack.append(left.intersection(right))
                else:
                    raise ValueError("Invalid boolean expression: not enough operands for AND operator")
            elif token == 'OR':
                if len(eval_stack) >= 2:
                    right = eval_stack.pop()
                    left = eval_stack.pop()
                    eval_stack.append(left.union(right))
                else:
                    raise ValueError("Invalid boolean expression: not enough operands for OR operator")
            elif token == 'NOT':
                if eval_stack:
                    operand = eval_stack.pop()
                    eval_stack.append(all_docs - operand)
                else:
                    raise ValueError("Invalid boolean expression: not enough operands for NOT operator")
            else:
                # It's a term, push the set of documents containing it
                eval_stack.append(self.inverted_index.get(token, set()))
        
        if len(eval_stack) == 1:
            return eval_stack[0]
        else:
            raise ValueError("Invalid boolean expression: incorrect number of terms and operators")

    def search(self, query):
        """
        Perform a boolean search using the given query.
        Returns a list of (doc_number, url) tuples for documents that match the query.
        """
        tokens = self._tokenize_query(query)
        try:
            result_doc_ids = self._evaluate_boolean_expression(tokens)
            return [(doc_id, self.doc_urls[doc_id]) for doc_id in result_doc_ids]
        except ValueError as e:
            print(f"Error: {e}")
            return []

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Boolean search engine for document collection")
    parser.add_argument("--input-dir", required=True, help="Directory containing the document collection")
    parser.add_argument("query", nargs='+', help="Boolean search query")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding the index even if it exists")
    
    args = parser.parse_args()
    
    # Join query arguments into a single string
    query = ' '.join(args.query)
    
    # Create the search engine
    search_engine = SearchEngine(args.input_dir)
    
    # Rebuild index if requested
    if args.rebuild_index:
        search_engine.build_index()
        search_engine.save_index()
    
    # Perform the search
    results = search_engine.search(query)
    
    # Print results
    print(f"Found {len(results)} results:")
    for doc_id, url in sorted(results):
        print(f"Document {doc_id}: {url}")

if __name__ == "__main__":
    main()
