# Search implementation using TF-IDF index

## Run

```bash
python vector_search.py --tfidf_dir output --index_file index.txt --query "some query for search" --use_lemmas
```

script can handle list of flags:

 - `--no-use_lemmas` - search using terms instead of lemmas
 - `--top_k` - Get more results


