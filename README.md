# CORD-19 Search Index System

A high-performance, memory-efficient Python pipeline designed to crawl, process, and index the massive CORD-19 (COVID-19 Open Research Dataset).

This project was built for the CS-250 Data Structures & Algorithms course at NUST (SE-15). It efficiently parses and indexes 50,000 priority papers out of the total 81,226 papers available in the TREC-COVID Round 1 dataset, creating a robust search foundation on standard hardware.

# **CORD-19 Search Index System**

A Python-based search indexing system for the COVID-19 Open Research Dataset (CORD-19). This system processes scientific research papers to build efficient search indexes that enable fast document retrieval and text analysis.

## Features

- **Memory-efficient streaming** - Processes compressed tar files without full extraction
- **Complete index generation** - Builds lexicon, forward, inverted, and backward indexes
- **Intelligent barreling** - Partitions indexes into 10 balanced barrels for parallel search
- **Advanced compression** - Uses VarByte encoding with mean-centered frequency compression
- **Scientific text processing** - Uses spaCy for advanced NLP and lemmatization
- **Multi-file support** - Handles all CORD-19 dataset subsets automatically
- **JSON output** - Generates standardized index files for easy integration

## Project Structure

```
cord19-search/
├── src/                    # Source code directory
│   ├── crawler.py         # Data streaming and text processing
│   ├── index.py           # Index generation and main pipeline
│   ├── barrel.py          # Barrel creation with load balancing
│   └── barrelled_index.py # Compression engine with VarByte encoding
├── data/                  # Data directory
│   ├── indexes/          # Generated index files
│   │   ├── lexicon.json
│   │   ├── forward_index.json
│   │   ├── inverted_index.json
│   │   └── backward_index.json
│   ├── barrels/          # Barrel partitions (10 balanced barrels)
│   │   ├── barrel_1.json
│   │   ├── barrel_2.json
│   │   ├── ...
│   │   ├── barrel_10.json
│   │   └── barrel_mappings.json
│   └── compressed_barrels/ # Compressed barrels (pickle format)
│       ├── compressed_barrel_1.pkl
│       ├── compressed_barrel_2.pkl
│       ├── ...
│       ├── compressed_barrel_10.pkl
│       └── doc_id_mapping.pkl
├── requirements.txt       # Python dependencies
├── README.md            # This file
```

## Output Files

| File | Description | Size | Format |
|------|-------------|------|--------|
| **lexicon.json** | Maps words to numeric IDs for efficient storage | ~20MB | `{"coronavirus": 1, "virus": 2, "infection": 3}` |
| **forward_index.json** | Tracks which word IDs appear in each document | ~200MB | `{"doc1": [1, 2, 3], "doc2": [2, 4, 5]}` |
| **inverted_index.json** | Search index - shows which documents contain each word with frequencies | ~300MB | `{"1": {"doc1": 2}, "2": {"doc1": 1, "doc2": 3}}` |
| **backward_index.json** | Stores actual words for each document (human-readable) | ~150MB | `{"doc1": ["coronavirus", "virus"], "doc2": ["patient", "treatment"]}` |
| **barrel_X.json** | Partitioned index segments (10 files) | ~30MB each | Contains subset of words with documents and frequencies |
| **barrel_mappings.json** | Routing tables for query processing | ~5MB | `{"word_to_barrel": {"covid": 3}, "doc_to_barrels": {"doc1": [1, 3]}}` |
| **compressed_barrel_X.pkl** | Compressed barrels with VarByte encoding | ~6MB each | Binary pickle files with 80-90% size reduction |
| **doc_id_mapping.pkl** | String to integer document ID mapping | ~2MB | Maps document IDs like "vho70jcx" to integers |

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdulbasitnikray-prog/cord19-crawler.git
   cd cord19-crawler
   ```

2. **Install dependencies**
   ```bash
   pip install spacy nltk pandas tqdm
   python -m spacy download en_core_web_sm
   ```

3. **Download CORD-19 dataset**
   - Get the dataset from [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
   - Place files in the directory structure expected by the system

## Usage

### Basic Indexing (Complete Pipeline)
```bash
python index.py                 # Step 1: Build indexes from dataset
python barrel.py                # Step 2: Create 10 balanced barrels
python barrelled_index.py       # Step 3: Apply compression to barrels
```

### Individual Steps
```python
# Step 1: Generate all indexes
from crawler import get_paper_stream
from index import generate_indexes_from_stream

paper_stream = get_paper_stream(max_papers=1000)
lexicon, forward_index, inverted_index, backward_index = generate_indexes_from_stream(paper_stream, max_papers=1000)

# Step 2: Create barrels
import barrel
barrels, doc_to_barrels, word_to_barrel = barrel.create_and_save_barrels(
    index_directory="indexes",
    output_dir="barrels",
    num_barrels=10
)

# Step 3: Compress barrels
import barrelled_index
barrelled_index.create_compressed_barrels(
    index_directory="indexes",
    barrel_dir="barrels",
    output_dir="compressed_barrels",
    num_barrels=10
)
```

## Configuration

Modify `BASE_PATH` in both `crawler.py` and `index.py` to point to your CORD-19 dataset location:

```python
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
```

Modify paths in `barrel.py` and `barrelled_index.py`:

```python
# barrel.py
index_directory = "indexes"  # Where index files are stored
output_dir = "barrels"       # Where to save barrel files

# barrelled_index.py
barrel_dir = "barrels"                # Input barrel directory
output_dir = "compressed_barrels"     # Output compressed barrels
```

## Performance

- **Processing Speed**: 5-7 papers per second
- **Memory Usage**: Minimal due to streaming architecture (~2GB RAM)
- **Dataset Scale**: Successfully handles 50,000+ papers from CORD-19 corpus
- **Output Size**: Efficient JSON storage with numeric encoding
- **Compression Ratio**: 80-90% storage reduction with VarByte encoding
- **Barreling**: 10 balanced partitions enable parallel search

## Use Cases

- **Research Search Engine** - Build custom search interfaces for COVID-19 literature
- **Text Analysis** - Analyze term frequencies and document similarities across barrels
- **Research Tools** - Integrate indexes into larger research platforms
- **Data Mining** - Extract insights from scientific paper collections
- **Educational Tool** - Study barreling and compression algorithms
- **Parallel Search** - Enable simultaneous searching across multiple barrels

## Future Enhancements

- Positional indexing for phrase search support
- TF-IDF scoring for relevance ranking
- Query expansion with medical synonyms
- Web-based search interface with barrel-level parallelization
- Real-time index updates with incremental barreling
- Distributed storage of barrels across multiple servers
- Advanced compression techniques (Golomb coding, Elias coding)
- Query optimization with barrel prioritization
- Caching layer for frequently accessed barrels


## Acknowledgments

- **CORD-19 Dataset**: Allen Institute for AI
- **Scientific NLP**: SciSpacy models
- **Course**: CS-250 Data Structures & Algorithms, NUST SE-15

## License

This project is designed for research use with the CORD-19 dataset. Please ensure compliance with the CORD-19 dataset license terms.
