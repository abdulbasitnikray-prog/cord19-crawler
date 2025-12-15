# CORD-19 Search Index System

A high-performance, memory-efficient Python pipeline designed to crawl, process, and index the massive CORD-19 (COVID-19 Open Research Dataset).

This project was built for the CS-250 Data Structures & Algorithms course at NUST (SE-15). It efficiently parses and indexes 50,000 priority papers out of the total 81,226 papers available in the TREC-COVID Round 1 dataset, creating a robust search foundation on standard hardware.

## Features

* **Memory-efficient streaming** - Processes compressed tar files without full extraction
* **Complete index generation** - Builds lexicon, forward, inverted, and backward indexes
* **Intelligent barreling** - Partitions indexes into 10 balanced barrels for parallel search
* **Advanced compression** - Uses VarByte encoding with mean-centered frequency compression
* **Scientific text processing** - Uses spaCy for advanced NLP and lemmatization
* **Semantic Analysis** - Includes trained Word2Vec models for context-aware query understanding
* **Autocomplete Engine** - Provides real-time search suggestions based on the indexed lexicon
* **Multi-file support** - Handles all CORD-19 dataset subsets automatically
* **JSON output** - Generates standardized index files for easy integration
* **Web Interface** - Flask-based search application with HTML templates
* **Single-word & Multi-word Search** - Supports both simple and complex query searches
* **Dynamic Indexing** - Add new documents to existing indexes without rebuilding
* **Barrel-based Parallel Search** - Leverages barrel partitions for concurrent query processing
* **Trie-based Optimization** - Fast barrel lookup with prefix-matching capabilities

## Project Structure

```
cord19-crawler/
├── data/                           # Data storage and indexes
│   ├── processed_corpus.csv        # Processed paper metadata with text
│   ├── barrels/                    # 10 balanced index partitions
│   │   ├── barrel_1.json
│   │   ├── barrel_2.json
│   │   └── ... (barrel_3 through barrel_10)
│   │   └── barrel_mappings.json    # Routing tables for query processing
│   ├── compressed_barrels/         # Compressed barrel versions (VarByte)
│   │   └── doc_id_mapping.pkl      # Document ID mappings
│   └── indexes/                    # Core index files
│       ├── lexicon.json            # Word ID mappings
│       ├── forward_index.json      # Word lists per document
│       ├── inverted_index.json     # Document lists per word
│       ├── backward_index.json     # Human-readable word lists
│       └── index_statistics.json   # Index metadata and statistics
├── src/                            # Source code directory
│   ├── __init__.py
│   ├── app.py                      # Flask web application
│   ├── autocomplete.py             # Search suggestion and completion engine
│   ├── barrel.py                   # Barrel creation with load balancing
│   ├── barreled_index.py           # Compression engine with VarByte encoding
│   ├── crawler.py                  # Data streaming and text processing
│   ├── dynamic_indexer.py          # Dynamic document addition engine
│   ├── index.py                    # Index generation and main pipeline
│   ├── multiwordSearch.py          # Multi-word query search with AND/OR logic
│   ├── preprocess_papers.py        # CSV preprocessing and corpus generation
│   ├── read_csv.py                 # Utility for parsing metadata CSVs
│   ├── semantic_search.py          # Semantic query expansion with Word2Vec
│   ├── singlewordSearch.py         # Single-word search with ranking
│   ├── train_semantic.py           # Training script for semantic models
│   ├── __pycache__/                # Python cache files
│   └── templates/                  # Web interface templates
│       ├── index.html              # Search interface
│       └── article.html            # Article detail view
├── cord19_semantic.model           # Trained Word2Vec semantic model
├── cord19_semantic.model.syn1neg.npy
├── cord19_semantic.model.wv.vectors.npy
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Output Files

| File | Description | Size | Format |
| :--- | :--- | :--- | :--- |
| **lexicon.json** | Maps words to numeric IDs for efficient storage | ~20MB | `{"coronavirus": 1, "virus": 2, "infection": 3}` |
| **forward_index.json** | Tracks which word IDs appear in each document | ~200MB | `{"doc1": [1, 2, 3], "doc2": [2, 4, 5]}` |
| **inverted_index.json** | Search index - shows which documents contain each word with frequencies | ~300MB | `{"1": {"doc1": 2}, "2": {"doc1": 1, "doc2": 3}}` |
| **backward_index.json** | Stores actual words for each document (human-readable) | ~150MB | `{"doc1": ["coronavirus", "virus"], "doc2": ["patient", "treatment"]}` |
| **barrel_X.json** | Partitioned index segments (10 files) | ~30MB each | Contains subset of words with documents and frequencies |
| **barrel_mappings.json** | Routing tables for query processing | ~5MB | `{"word_to_barrel": {"covid": 3}, "doc_to_barrels": {"doc1": [1, 3]}}` |
| **compressed_barrel_X.pkl** | Compressed barrels with VarByte encoding | ~6MB each | Binary pickle files with 80-90% size reduction |
| **doc_id_mapping.pkl** | Document ID to path mapping | ~1MB | Pickle file for document resolution |
| **barrel_trie.pkl** | Trie data structure for fast barrel lookup | ~2MB | Serialized trie for prefix matching |
| **processed_corpus.csv** | Processed paper metadata with full text | ~500MB | CSV format: doc_id, title, authors, text |
| **cord19_semantic.model** | Trained Word2Vec/Semantic model | ~50MB | Binary model data |
| **index_statistics.json** | Metadata about index composition | ~100KB | Statistics and index metadata |

## Installation

1.  Clone the repository

    ```bash
    git clone [https://github.com/abdulbasitnikray-prog/cord19-crawler.git](https://github.com/abdulbasitnikray-prog/cord19-crawler.git)
    cd cord19-crawler
    ```

2.  Install dependencies

    ```bash
    pip install spacy nltk pandas tqdm gensim
    python -m spacy download en_core_web_sm
    ```

3. **Download CORD-19 dataset**
   - Get the dataset from [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
   - Place files in the directory structure expected by the system

## Usage

### Web Application (Recommended)

Start the Flask web server for interactive searching:

```bash
python src/app.py
```

Then open your browser to `http://localhost:5000` and use the search interface.

### Basic Indexing (Complete Pipeline)

```bash
python src/index.py           # Step 1: Build indexes from dataset
python src/barrel.py          # Step 2: Create 10 balanced barrels
python src/barreled_index.py  # Step 3: Apply compression to barrels
python src/preprocess_papers.py # Step 4: Generate corpus metadata
```

### Semantic Training

To train the semantic model on the new dataset:

```bash
python src/train_semantic.py
```

### Search Types

**Single-word Search:**

```python
from src.singlewordSearch import search_word

results = search_word("coronavirus", max_results=10)
# Returns: List of documents containing the word with scores
```

**Multi-word Search:**

```python
from src.multiwordSearch import multi_word_search

results = multi_word_search(["vaccine", "treatment"], operator="AND")
# Returns: Documents containing all specified words
```

**Semantic Search:**

```python
from src.semantic_search import semantic_engine

semantic_engine.load_model()
similar_words = semantic_engine.get_similar_words("virus", top_n=5)
# Returns: Semantically similar words for query expansion
```

**Autocomplete:**

```python
from src.autocomplete import AutocompleteEngine

suggestions = AutocompleteEngine.get_suggestions("coro", max_suggestions=5)
# Returns: Search suggestions matching prefix
```

### Dynamic Indexing

Add new documents to existing indexes without rebuilding:

```python
from src.dynamic_indexer import DynamicIndexer

indexer = DynamicIndexer(index_dir="data/indexes")
indexer.load_indexes()
indexer.add_document(doc_id="new_doc", content="Paper content here...")
indexer.save_indexes()
```

### Individual Steps (Advanced)

**Step 1: Generate all indexes**

```python
from src.crawler import get_paper_stream
from src.index import generate_indexes_from_stream

paper_stream = get_paper_stream(max_papers=1000)
lexicon, forward_index, inverted_index, backward_index = generate_indexes_from_stream(paper_stream, max_papers=1000)
```

**Step 2: Create barrels**

```python
import src.barrel as barrel
barrels, doc_to_barrels, word_to_barrel = barrel.create_and_save_barrels(
    index_directory="indexes",
    output_dir="barrels",
    num_barrels=10
)
```

**Step 3: Compress barrels**

```python
import src.barreled_index as barreled_index
barreled_index.create_compressed_barrels(
    index_directory="indexes",
    barrel_dir="barrels",
    output_dir="compressed_barrels",
    num_barrels=10
)
```

## Configuration

Modify `BASE_PATH` in `src/crawler.py` and `src/index.py` to point to your CORD-19 dataset location:

```python
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
```

## Performance

  * **Processing Speed:** 5-7 papers per second
  * **Memory Usage:** Minimal due to streaming architecture (\~2GB RAM)
  * **Dataset Scale:** Successfully handles 50,000+ papers from CORD-19 corpus
  * **Output Size:** Efficient JSON storage with numeric encoding
  * **Compression Ratio:** 80-90% storage reduction with VarByte encoding
  * **Barreling:** 10 balanced partitions enable parallel search

## Use Cases

- **Research Search Engine** - Build custom search interfaces for COVID-19 literature
- **Text Analysis** - Analyze term frequencies and document similarities across barrels
- **Research Tools** - Integrate indexes into larger research platforms
- **Data Mining** - Extract insights from scientific paper collections
- **Educational Tool** - Study barreling, compression, and semantic search algorithms
- **Parallel Search** - Enable simultaneous searching across multiple barrels
- **Document Management** - Dynamically add and index new research papers
- **Semantic Query Expansion** - Improve search relevance with context-aware suggestions
- **Web Application** - Deploy as a full-featured research paper search platform

## Future Enhancements

  * Positional indexing for phrase search support
  * TF-IDF scoring for relevance ranking
  * Query expansion with medical synonyms
  * Web-based search interface improvements
  * Real-time index updates with incremental barreling
  * Distributed storage of barrels across multiple servers
  * Advanced filtering by paper metadata (date, author, journal)
  * Integration with citation graph analysis
  * Batch processing and bulk document import
  * Performance optimization for sub-second query latency

## Acknowledgments

  * **CORD-19 Dataset:** Allen Institute for AI
  * **Scientific NLP:** SciSpacy models
  * **Course:** CS-250 Data Structures & Algorithms, NUST SE-15

## License

This project is designed for research use with the CORD-19 dataset. Please ensure compliance with the CORD-19 dataset license terms.

```
```
