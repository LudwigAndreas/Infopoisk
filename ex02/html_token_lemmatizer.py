#!/usr/bin/env python3
import os
import re
import sys
import argparse
from typing import List, Dict, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class HTMLTokenizer:
    def __init__(self):
        """
        Initialize the tokenizer with necessary resources
        """
        # Comprehensive set of stopwords and additional filter words
        self.stop_words = set(stopwords.words('english') + [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 
            'into', 'over', 'after', 'beneath', 'under', 'above'
        ])
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_html(self, html_content: str) -> str:
        """
        Preprocess HTML content by removing scripts, styles, and extracting text
        
        Args:
            html_content (str): Raw HTML content
        
        Returns:
            str: Extracted text from HTML
        """
        try:
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ')
            
            return text
        except Exception as e:
            print(f"Error preprocessing HTML: {e}")
            return ""

    def is_valid_token(self, token: str) -> bool:
        """
        Check if a token is valid
        
        Args:
            token (str): Token to validate
        
        Returns:
            bool: Whether the token is valid
        """
        return (
            len(token) > 1 and  # More than 1 character
            token.isalpha() and  # Only alphabetic
            token.lower() not in self.stop_words  # Not a stopword
        )

    def advanced_lemmatize(self, token: str) -> str:
        """
        Perform advanced lemmatization with multiple attempts
        
        Args:
            token (str): Token to lemmatize
        
        Returns:
            str: Lemmatized token
        """
        # Try different lemmatization approaches
        lemma_attempts = [
            self.lemmatizer.lemmatize(token),           # default noun form
            self.lemmatizer.lemmatize(token, pos='v'),  # verb form
            self.lemmatizer.lemmatize(token, pos='a'),  # adjective form
            self.lemmatizer.lemmatize(token, pos='r')   # adverb form
        ]
        
        # Find the shortest lemma (usually the most base form)
        return min(set(lemma_attempts), key=len)

    def tokenize_file(self, file_path: str) -> Tuple[Set[str], Dict[str, List[str]]]:
        """
        Tokenize and lemmatize a single HTML file
        
        Args:
            file_path (str): Path to the HTML file
        
        Returns:
            tuple: (unique tokens, lemma groups)
        """
        # Set to store unique tokens
        unique_tokens = set()
        
        # Dictionary to store lemma groupings
        lemma_groups: Dict[str, List[str]] = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract text from HTML
            text = self.preprocess_html(html_content)
            
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Filter and process tokens
            for token in tokens:
                if self.is_valid_token(token):
                    unique_tokens.add(token)
                    
                    # Advanced lemmatization
                    lemma = self.advanced_lemmatize(token)
                    
                    # Group by lemma
                    if lemma not in lemma_groups:
                        lemma_groups[lemma] = []
                    if token not in lemma_groups[lemma]:
                        lemma_groups[lemma].append(token)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        return unique_tokens, lemma_groups

    def write_output(self, unique_tokens: Set[str], lemma_groups: Dict[str, List[str]], 
                     tokens_output: str, lemmas_output: str):
        """
        Write tokenization and lemmatization results to files
        
        Args:
            unique_tokens (Set[str]): Set of unique tokens
            lemma_groups (Dict[str, List[str]]): Grouped lemmas
            tokens_output (str): Path to tokens output file
            lemmas_output (str): Path to lemmas output file
        """
        # Write unique tokens
        with open(tokens_output, 'w', encoding='utf-8') as f:
            for token in sorted(unique_tokens):
                f.write(f"{token}\n")
        
        # Write lemmatized tokens
        with open(lemmas_output, 'w', encoding='utf-8') as f:
            for lemma, token_list in sorted(lemma_groups.items()):
                # Sort tokens and remove duplicates while preserving order
                unique_tokens = list(dict.fromkeys(token_list))
                # unique_tokens.append(lemma)
                f.write(f"{' '.join(sorted(unique_tokens))}\n")
        
        print(f"Processed {len(unique_tokens)} unique tokens")
        print(f"Grouped into {len(lemma_groups)} lemmas")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tokenize and lemmatize HTML files')
    parser.add_argument('input_dir', help='Directory containing HTML files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create tokenizer instance
    tokenizer = HTMLTokenizer()
    
    # Process files
    for filename in os.listdir(args.input_dir):
        # Check if filename matches {number}.html pattern
        match = re.match(r'^(\d+)\.html$', filename)
        if not match:
            continue
        
        # Get the file number
        file_number = match.group(1)
        
        # Construct full file path
        file_path = os.path.join(args.input_dir, filename)
        
        # Generate output file paths
        tokens_output = os.path.join(args.input_dir, f'tokens_{file_number}.txt')
        lemmas_output = os.path.join(args.input_dir, f'lemmas_{file_number}.txt')
        
        # Tokenize the file
        unique_tokens, lemma_groups = tokenizer.tokenize_file(file_path)
        
        # Write output files
        tokenizer.write_output(unique_tokens, lemma_groups, 
                               tokens_output, lemmas_output)

if __name__ == '__main__':
    main()
