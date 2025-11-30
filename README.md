# CORD-19 Search Index System

A Python-based search indexing system for the COVID-19 Open Research Dataset (CORD-19). This system processes scientific research papers to build efficient search indexes that enable fast document retrieval and text analysis.

## Features

- **Memory-efficient streaming** - Processes compressed tar files without full extraction
- **Complete index generation** - Builds lexicon, forward, inverted, and backward indexes
- **Scientific text processing** - Uses spaCy for advanced NLP and lemmatization
- **Multi-file support** - Handles all CORD-19 dataset subsets automatically
- **JSON output** - Generates standardized index files for easy integration

## Project Structure

```
cord19-search/
├── crawler.py          # Data streaming and text processing
├── index.py            # Index generation and main pipeline
├── lexicon.json        # Generated vocabulary mapping
├── forward_index.json  # Document-to-words mapping
├── inverted_index.json # Word-to-documents mapping (search index)
└── backward_index.json # Human-readable document contents
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cord19-search
   ```

2. **Install dependencies**
   ```bash
   pip install spacy nltk pandas
   python -m spacy download en_core_web_sm
   ```

3. **Download CORD-19 dataset**
   - Get the dataset from [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
   - Place files in the directory structure expected by the system

## Usage

### Basic Indexing
```bash
python index.py
```

This will:
- Process all CORD-19 tar files (`biorxiv_medrxiv.tar.gz`, `comm_use_subset.tar.gz`, etc.)
- Extract and clean text from research papers
- Build all search indexes
- Save results as JSON files

### Custom Processing
```python
from crawler import get_paper_stream
from index import generate_indexes_from_stream

# Process specific number of papers
paper_stream = get_paper_stream(max_papers=1000)
lexicon, forward_index, inverted_index, backward_index = generate_indexes_from_stream(paper_stream, max_papers=1000)
```

## Output Files

| File | Description | Format Example |
|------|-------------|----------------|
| **lexicon.json** | Maps words to numeric IDs for efficient storage | `{"coronavirus": 1, "virus": 2, "infection": 3}` |
| **forward_index.json** | Tracks which word IDs appear in each document | `{"doc1": [1, 2, 3], "doc2": [2, 4, 5]}` |
| **inverted_index.json** | Search index - shows which documents contain each word | `{"1": {"doc1": 2}, "2": {"doc1": 1, "doc2": 3}}` |
| **backward_index.json** | Stores actual words for each document (human-readable) | `{"doc1": ["coronavirus", "virus"], "doc2": ["patient", "treatment"]}` |


## Configuration

Modify `BASE_PATH` in both `crawler.py` and `index.py` to point to your CORD-19 dataset location:

```python
BASE_PATH = "path/to/your/cord-19/dataset"
```

## Performance

- **Processing Speed**: 5-7 papers per second
- **Memory Usage**: Minimal due to streaming architecture
- **Dataset Scale**: Successfully handles full CORD-19 corpus
- **Output Size**: Efficient JSON storage with numeric encoding

## Use Cases

- **Research Search Engine** - Build custom search interfaces for COVID-19 literature
- **Text Analysis** - Analyze term frequencies and document similarities
- **Research Tools** - Integrate indexes into larger research platforms
- **Data Mining** - Extract insights from scientific paper collections

## Future Enhancements

- Positional indexing for phrase search support
- TF-IDF scoring for relevance ranking
- Query expansion with medical synonyms
- Web-based search interface
- Real-time index updates

## License

This project is designed for research use with the CORD-19 dataset. Please ensure compliance with the CORD-19 dataset license terms.
