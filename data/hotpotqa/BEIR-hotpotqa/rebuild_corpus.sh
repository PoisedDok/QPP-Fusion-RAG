#!/bin/bash
# Rebuild corpus.jsonl from split parts

if [ -f "corpus.jsonl" ]; then
    echo "corpus.jsonl already exists"
    exit 0
fi

echo "Rebuilding corpus.jsonl from parts..."
cat corpus_part_* > corpus.jsonl
echo "Done. Corpus size: $(du -h corpus.jsonl | cut -f1)"

