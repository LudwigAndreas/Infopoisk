# HTML Token Lemmatizer

## Overview
This Python script processes HTML files in a directory, extracting and processing tokens with the following features:
- Tokenization of HTML content
- Removal of duplicates, conjunctions, prepositions, and numbers
- Cleaning of "garbage" tokens (words with mixed letters and numbers)
- Lemmatization of tokens

## Prerequisites
- Python 3.7+
- Install required libraries:
  ```
  pip install -r requirements.txt
  ```

## Setup
1. Place your HTML files in the directory
2. Name the HTML files numerically (1.html, 2.html, etc.)

## Usage
Run the script:
```
python html_token_lemmatizer.py input_directory
```

## Notes
- The script uses NLTK for advanced text processing
- Tokens are filtered to remove:
  * Short tokens
  * Tokens with numbers
  * Stopwords
  * Non-alphabetic tokens

## Dependencies
- beautifulsoup4
- nltk
