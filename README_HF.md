---
title: CORD-19 Search Engine
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# CORD-19 Search Engine

A high-performance medical research paper search engine for the COVID-19 Open Research Dataset (CORD-19).

## Features

- **Fast Multi-Word Search**: TF-IDF based ranking with <1.5s query time
- **Semantic Search**: Word2Vec-powered query expansion for better results
- **Autocomplete**: Real-time suggestions from 363K+ terms
- **Dynamic Indexing**: Upload new papers on-the-fly
- **Optimized Performance**: ~45% memory reduction, 35-40% faster searches

## Usage

1. Enter your search query in the search bar
2. Toggle "Semantic Search" for synonym expansion
3. Click on results to view full paper content
4. Upload new papers via the upload interface

## Technical Details

- **Dataset**: 51,045 COVID-19 research papers
- **Search Algorithm**: TF-IDF with barreled indexing
- **Compression**: VarByte encoding for space efficiency
- **Technology Stack**: Flask, Python, spaCy, Gensim

Built with ❤️ for medical research accessibility.
