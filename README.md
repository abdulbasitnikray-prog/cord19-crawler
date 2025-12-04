# CORD-19 Indexing System

## Files:
- `src/posting_barrel_demo.py` - Complete demo with sample paper
- `src/posting_barrel_full.py` - Full version for actual dataset  
- `src/posting_list.py` - Core posting list functions
- `src/partition_lexicon.py` - Core barrel partitioning functions
- `data/sample_paper.json` - Sample paper for testing

## Quick Start:
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (uses embedded sample paper)
python src/posting_barrel_demo.py

# Run full version (uses your CORD-19 dataset)
# First edit BASE_PATH in posting_barrel_full.py
python src/posting_barrel_full.py