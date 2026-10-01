# Umapper

A Python tool for analyzing and comparing the language used across YouTube channels. The program processes video transcripts, extracts linguistic and semantic features, and visualizes similarities between channels.

Advanced version is coming soon as: https://github.com/Incodi/Centroids/

## What It Does

The tool processes transcript data for selected YouTube channels and:

Generates semantic embeddings using Sentence-BERT

Measures lexical diversity using MTLD

Estimates text complexity using LZMA compression ratios

Uses UMAP to reduce channel embeddings to two dimensions

Groups similar channels using K-Means clustering

Creates visualizations based on semantic similarity, MTLD, or compression ratio


Videos can also be filtered by upload date, duration, live-stream status, and transcript length.

## Technologies

Python, Sentence Transformers, scikit-learn, UMAP, NumPy, Matplotlib, LZMA, and LexicalRichness.

## Example Use case

python3 src/mapper.py animalogic bogxd mkbhd eons \
  --date-from 20200101 \
  --duration-from 8:00 \
  --exclude-live \
  --token-limit 400000 \
  --min-words 400000 \
  --compression-coloring

The resulting visualization shows how the selected channels relate to one another based on their transcript language and can optionally highlight differences in lexical diversity or compression ratio.
