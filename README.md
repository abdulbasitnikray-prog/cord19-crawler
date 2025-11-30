# CORD-19 Search Engine Indexer:

A high-performance, memory-efficient Python pipeline designed to crawl, process, and index the massive **CORD-19 (COVID-19 Open Research Dataset)**.

This project was built for the **CS-250 Data Structures & Algorithms** course at **NUST (SE-15)**. It efficiently parses and indexes **50,000 priority papers** out of the total **81,226 papers** available in the TREC-COVID Round 1 dataset, creating a robust search foundation on standard hardware.

## Key Features

* **Streaming Pipeline:** Uses Python generators (`yield`) to stream data directly from the compressed `.tar.gz` archive. This eliminates the need to extract 100GB+ of JSON files to the hard drive and prevents RAM crashes.
* **Single-Pass NLP:** optimized text processing pipeline that performs cleaning, tokenization, stop-word removal, and lemmatization in a single pass using **SciSpacy**.
* **Memory Safe:** Processes documents sequentially, ensuring memory usage stays flat (~2GB) regardless of dataset size.
* **Triple Index Generation:** Simultaneously builds the three core structures required for a search engine:
    1.  **Lexicon** (Word $\rightarrow$ ID)
    2.  **Forward Index** (Document $\rightarrow$ [Word IDs])
    3.  **Inverted Index** (Word ID $\rightarrow$ {Document: Frequency})

## Directory Structure

Here is an explanation of the files currently in the repository:

| File / Folder | Description |
| :--- | :--- |
| **`Crawler.py`** | The main engine script. It handles the TAR streaming, text preprocessing (SciSpacy), and the generation of all three indices. |
| **`read_csv.py`** | A utility script used to inspect and visualize the `metadata.csv` file to understand the dataset structure before processing. |
| **`extraction.py`** | A utility script to perform "surgical extraction" of sample JSON papers from the massive TAR archive without unzipping the whole thing. |
| **`sample_data/`** | A folder containing a small batch of extracted JSON papers for testing the code without the full dataset. |
| **`Forward_Inverted_Lexicon.001`** | Split archive (Part 1). Contains the generated `lexicon.json`, `forward_index.json`, and `inverted_index.json`. Split to bypass GitHub's file size limits. |
| **`Forward_Inverted_Lexicon.002`** | Split archive (Part 2). The second part of the zipped indices. |
| **`.gitignore`** | Configuration to prevent uploading massive raw datasets (like `document_parses.tar.gz`) or local environment folders to GitHub. |

## Prerequisites

* **Python 3.9+** (Recommended for library compatibility on Windows)
* **RAM:** 4GB minimum
* **Disk Space:** Enough to hold the `document_parses.tar.gz` (~10-15GB) and the output JSONs (~300MB).

## Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/cord19-indexer.git](https://github.com/your-username/cord19-indexer.git)
    cd cord19-indexer
    ```

2.  **Set up a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Note: We use specific versions to ensure compatibility with the `scispacy` scientific model.
    ```bash
    pip install "numpy<2.0.0"
    pip install blis==0.7.11 spacy==3.7.5
    pip install scispacy
    pip install [https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz)
    ```

## Configuration

Open `Crawler.py` and modify the configuration block at the top to match your local paths:

```python
# Path containing metadata.csv and document_parses.tar.gz
BASE_PATH = "D:/Cord19/cord/2022" 

# Number of papers to process (50,000 satisfies project requirements)
BATCH_SIZE = 50000
```
## Usage

Run the crawler script:

```python
python Crawler.py
```
## Execution Flow

**Metadata Loading:** Loads the CSV map into memory ($O(1)$ lookup).

**Streaming:** Opens the TAR file and processes valid JSONs one by one.

**Processing:** Extracts the first 35 lines (Abstract/Intro) to ensure relevance.

* Cleans text using pre-compiled Regex.

* Lemmatizes using the Scientific AI Model.

**Indexing:** Updates the Lexicon, Forward Index, and Inverted Index in real-time.

**Saving:** Dumps the final data structures to JSON files.

## Output Files

The script generates three files in your working directory:

| File / Folder | Description | Approx Size |
| :--- | :--- | :--- |
| **`lexicon.json`** | Dictionary mapping unique words strings to integer IDs. | ~ 20 MB
| **`forward_index.json`** |Map of Document IDs to a list of word IDs found in that doc. | ~ 200 MB
| **`inverted_index.json`** | Map of Word IDs to a dictionary of Document IDs and their term frequency. | ~ 300 MB

## License
This project processes data from the [CORD-19 Dataset](https://github.com/allenai/cord19), provided by the Allen Institute for AI.
