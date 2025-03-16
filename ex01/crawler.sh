#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file_with_links>"
    exit 1
fi

# Input file with links
INPUT_FILE="$1"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

# Output directory
OUTPUT_DIR="downloaded_pages"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Index file
INDEX_FILE="$OUTPUT_DIR/index.txt"

# Create or clear index file
> "$INDEX_FILE"

# Counter for file numbering
COUNTER=1

# Download each URL
echo "Starting downloads..."
while read -r URL; do
    # Skip empty lines
    if [ -z "$URL" ]; then
        continue
    fi
    
    # Output filename
    OUTPUT_FILE="$OUTPUT_DIR/${COUNTER}.html"
    
    echo "Downloading $URL to $OUTPUT_FILE"
    
    # Download only HTML content (no scripts, css, etc.)
    # wget options:
    # -q: quiet mode
    # -O: output to file
    # --no-check-certificate: don't check SSL certificates
    # --timeout=30: timeout after 30 seconds
    # --tries=3: retry 3 times
    # --no-directories: don't create directories
    wget -q \
         --no-check-certificate \
         --timeout=30 \
         --tries=3 \
         --reject "*.css,*.js,*.png,*.jpg,*.jpeg,*.gif,*.svg,*.ico,*.woff,*.woff2,*.ttf,*.eot,*.otf" \
         -O "$OUTPUT_FILE" \
         "$URL"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        # Add entry to index file
        echo "$COUNTER $URL" >> "$INDEX_FILE"
        echo "Downloaded successfully."
    else
        echo "Failed to download $URL"
        rm -f "$OUTPUT_FILE"  # Remove empty file
        # Don't increment counter for failed downloads
        continue
    fi
    
    # Increment counter
    ((COUNTER++))
    
done < "$INPUT_FILE"

echo "Download complete. Downloaded $(($COUNTER - 1)) pages."
echo "Files saved in $OUTPUT_DIR"
echo "Index file created at $INDEX_FILE"
