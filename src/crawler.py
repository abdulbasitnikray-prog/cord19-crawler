# crawler.py - Memory-optimized for parallel batch processing with scispaCy
import json as js 
import os
import csv
import tarfile
import time
import spacy 
import re
import string
from collections import defaultdict

# Base paths
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
EXTRACTION_FOLDER = os.path.join(BASE_PATH, "document_parses")

# Global constraints preventing repetitive compile of regex patterns
PUNCT_PATTERN = re.compile(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')))
SPACE_PATTERN = re.compile(r'\s+')
DIGIT_PATTERN = re.compile(r'\d+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
# Pattern to detect gibberish (like if there's multiple nonalphabetic characters in a row)
GIBBERISH_PATTERN = re.compile(r'[^a-zA-Z\s]{3,}')
# Pattern for chemical/biological sequences (like DNA/protein sequences)
SEQUENCE_PATTERN = re.compile(r'\b[ACGTU]{6,}\b|\b[ACGTU]{3,}(?:[ACGTU]{3,})+\b', re.IGNORECASE)

# Global NLP model for worker processes
_nlp_model = None

def init_worker_nlp():
    """Initialize lightweight ScispaCy model with memory optimization (the other one was too heavy and took too long)"""
    global _nlp_model
    if _nlp_model is None:
        # we only need tokenizer, tagger, lemmatizer (minimise components)
        disable_pipes = ["parser", "ner", "textcat", "attribute_ruler", "senter"]
        
        try:
            _nlp_model = spacy.load("en_core_sci_sm", disable=disable_pipes)
            _nlp_model.max_length = 1000000  # 1 million characters
            
        except Exception as e:
            print(f"worker failed to load scispaCy model: {e}")
            print("going to fallback to web_sm model")
            try:
                # code to fallback to web_sm if sci_sm fails (web-sm is smaller but less accurate)
                _nlp_model = spacy.load("en_core_web_sm", disable=disable_pipes)
                _nlp_model.max_length = 1000000
                # print("worker: Loaded en_core_web_sm (fallback)")
            except:
                print("could not load any spaCy model")
                _nlp_model = None

def get_scipacy_model():
    """Get spaCy model for text processing (main thread) - optimised for memory"""
    # Use same configuration as init_worker_nlp
    disable_pipes = ["parser", "ner", "textcat", "attribute_ruler", "senter"]
    
    try:
        # Load without invalid config
        nlp = spacy.load("en_core_sci_sm", disable=disable_pipes)
        nlp.max_length = 1000000
        return nlp
    except Exception as e:
        try:
            nlp = spacy.load("en_core_web_sm", disable=disable_pipes)
            nlp.max_length = 1000000
            return nlp
        except:
            print(f"An exception occurred: {e}")
            return None

def clean_text(lines):
    """clean text with memory-safe preprocessing"""
    if not lines: 
        return ""
    
    # join lines and convert to lowercase
    full_text = " ".join(lines).lower()
    
    # truncating very long texts to prevent memory issues
    max_chars = 50000  # limit to 50K characters
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    
    # no urls
    full_text = URL_PATTERN.sub('', full_text)
    # no emails
    full_text = EMAIL_PATTERN.sub('', full_text)
    # no chemical/biological sequences (DNA, protein sequences)
    full_text = SEQUENCE_PATTERN.sub('', full_text)
    # no gibberish (multiple non-alphabetic characters)
    full_text = GIBBERISH_PATTERN.sub('', full_text)
    # no digits
    full_text = DIGIT_PATTERN.sub('', full_text)
    # no problematic punctuation (keeping some for sentence structure)
    full_text = PUNCT_PATTERN.sub('', full_text)
    # no extra whitespace
    full_text = SPACE_PATTERN.sub(' ', full_text)
    # removing standalone single letters (except 'a' and 'i')
    words = full_text.split()
    filtered_words = []
    for word in words:
        if len(word) == 1 and word not in ['a', 'i']:
            continue
        if len(word) > 50:  # skip extremely long words (not really useful)
            continue
        filtered_words.append(word)
    
    full_text = ' '.join(filtered_words)
    
    return full_text.strip()

def extract_text(json_parse, max_lines=50):
    """Extract text from paper JSON with character limits"""
    if json_parse is None:
        return []
    
    lines = []
    total_chars = 0
    
    # extract title if available
    if 'metadata' in json_parse and 'title' in json_parse['metadata']:
        title = json_parse['metadata']['title']
        if title and isinstance(title, str):
            lines.extend(title.splitlines())
            total_chars += len(title)
    
    # extract abstract
    if 'abstract' in json_parse:
        for entry in json_parse['abstract']:
            text = entry.get("text", "")
            if text and isinstance(text, str):
                lines.extend(text.splitlines())
                total_chars += len(text)
    
    # extract body text with character limit
    if 'body_text' in json_parse:
        for section in json_parse['body_text']:
            text = section.get("text", "")
            if text and isinstance(text, str):
                # skip if already too much text
                if total_chars > 50000:  # 50K character limit
                    break
                lines.extend(text.splitlines())
                total_chars += len(text)
                if len(lines) >= max_lines:  
                    break
    
    # take only unique lines to reduce repetition
    unique_lines = []
    seen_lines = set()
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and line_stripped not in seen_lines:
            seen_lines.add(line_stripped)
            unique_lines.append(line_stripped)
    
    return unique_lines[:max_lines]

def process_paper_single(json_parse, cord_uid=None):  
    """Process single paper - text parsing only (for single-threaded use)"""
    if json_parse is None:
        return None
    raw_lines = extract_text(json_parse)
    if not raw_lines: 
        return None
    full_text = clean_text(raw_lines)
    if not full_text: 
        return None
    
    nlp = get_scipacy_model()
    if not nlp:
        return None
        
    doc = nlp(full_text) 
    indexed_tokens = []
    for token in doc:
        if (token.is_stop or token.is_punct or token.is_space or 
            token.like_num or len(token.text) < 2):
            continue
        token_data = {
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_
        }
        indexed_tokens.append(token_data)
    return {"tokens": indexed_tokens}

def process_paper_batch(batch):
    """
    Process a batch of papers using spaCy's pipe for efficiency
    Returns: List of processed papers with tokens
    """
    global _nlp_model
    
    if _nlp_model is None:
        init_worker_nlp()
        if _nlp_model is None:
            return []
    
    papers_data = []
    texts = []
    
    # extract and clean texts
    for paper in batch:
        raw_lines = extract_text(paper.get('json_parse'))
        if raw_lines:
            full_text = clean_text(raw_lines)
            if full_text and len(full_text) > 10:  # Skip very short texts
                texts.append(full_text)
                papers_data.append({
                    "cord_uid": paper.get("cord_uid"),
                    "title": paper.get("title", "")
                })
    
    if not texts:
        return []
    
    # batch process with spaCy pipe - with smaller batch size for memory
    results = []
    batch_size = min(20, len(texts))  # SMALLER batch size for memory safety
    
    try:
        for doc in _nlp_model.pipe(texts, batch_size=batch_size, n_process=1):
            if not papers_data:
                continue
                
            paper_info = papers_data.pop(0)
            tokens = []
            
            for token in doc:
                if (not token.is_stop and not token.is_punct and 
                    not token.is_space and not token.like_num and 
                    len(token.text) >= 2 and token.pos_ != 'X'):  # skip "other" category
                    
                    # Store token with POS information
                    token_info = {
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'tag': token.tag_
                    }
                    tokens.append(token_info)
            
            if tokens:
                results.append({
                    "cord_uid": paper_info["cord_uid"],
                    "tokens": tokens  # pos info for queries
                })
    
    except Exception as e:
        print(f"  Worker error (skipping batch): {str(e)[:100]}")
        return []
    
    return results

def stream_tar_dataset(metadata_path, tar_path, max_papers=None):
    """
    Reads directly from the .tar file without extracting it.
    Matches files in the TAR to entries in metadata.csv.
    Yields papers one by one for memory efficiency.
    """
    print("Step 1: Loading metadata into memory for fast lookup...")
    meta_lookup = {}
    
    # 1. Load metadata into a dictionary: { "sha_hash": {row_data} }
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We map the SHA (filename identifier) to the row data
                if row['sha']: 
                    # specific fix for multiple shas in one field
                    for single_sha in row['sha'].split(';'):
                        meta_lookup[single_sha.strip()] = row
                # We can also map pmcid if needed, but sha is primary for pdf_json
                if row['pmcid']:
                    meta_lookup[row['pmcid']] = row
                    
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    print(f"Loaded metadata for {len(meta_lookup)} papers.")
    print("Step 2: Streaming through the TAR file...")
    
    found_count = 0
    
    # 2. Open the TAR file as a stream
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                # Stop if we hit the limit
                if max_papers and found_count >= max_papers:
                    break
                
                # We only care about .json files
                if not member.isfile() or not member.name.endswith('.json'):
                    continue
                
                # Extract the filename (SHA) from the path
                filename = os.path.basename(member.name).replace('.json', '').replace('.xml', '')
                
                # 3. Check if this file exists in our metadata
                if filename in meta_lookup:
                    meta_row = meta_lookup[filename]
                    
                    # Read the JSON file directly from the TAR stream
                    f = tar.extractfile(member)
                    if f:
                        try:
                            content = js.load(f)
                            yield {
                                "cord_uid": meta_row["cord_uid"],
                                "title": meta_row["title"],
                                "json_parse": content
                            }
                            found_count += 1
                            if found_count % 5000 == 0:
                                print(f"  Streamed {found_count:,} papers...")
                        except:
                            pass # Skip malformed JSONs
                            
    except FileNotFoundError:
        print(f"Error: Could not find the tar file at {tar_path}")
        return
    except GeneratorExit:
        # Handle generator being closed early
        return

    print(f"Successfully streamed {found_count:,} papers from archive.")

def get_paper_stream(max_papers=None):
    """
    Returns a generator that yields papers one by one from all tar files.
    """
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: Could not find metadata.csv at: {metadata_path}")
        return None

    # List of all tar files to process
    tar_files = [
        "biorxiv_medrxiv.tar.gz",
        "comm_use_subset.tar.gz", 
        "custom_license.tar.gz",
        "noncomm_use_subset.tar.gz"
    ]
    
    print("Creating paper stream generator from multiple tar files...")
    
    total_found = 0
    for tar_filename in tar_files:
        tar_path = os.path.join(BASE_PATH, tar_filename)
        
        if not os.path.exists(tar_path):
            print(f"Warning: Could not find {tar_filename}")
            continue
            
        print(f"\nProcessing {tar_filename}...")
        
        # Get papers from this tar file
        for paper in stream_tar_dataset(metadata_path, tar_path, max_papers=None):
            if max_papers and total_found >= max_papers:
                print(f"Reached max papers limit: {max_papers:,}")
                return
            total_found += 1
            yield paper
    
    print(f"\nTotal papers available: {total_found:,}")

def get_paper_batches(batch_size=100, max_papers=None):
    """
    Yields batches of papers for parallel processing
    Returns: Generator of paper batches
    """
    print(f"Creating paper batches (batch_size={batch_size})...")
    
    paper_stream = get_paper_stream(max_papers)
    if not paper_stream:
        return
    
    current_batch = []
    batch_count = 0
    
    for paper in paper_stream:
        current_batch.append(paper)
        
        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []
            batch_count += 1
            
    
    # Yield final batch
    if current_batch:
        yield current_batch
        batch_count += 1
    
    print(f"Total batches created: {batch_count}")

def check_files():
    """Utility function to check if required files exist"""
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    
    print(f"Looking for metadata at: {metadata_path}")
    print(f"Exists: {os.path.exists(metadata_path)}")
    
    # List all files in BASE_PATH to see what's there
    if os.path.exists(BASE_PATH):
        print("\nFiles in directory:")
        tar_files = []
        for file in os.listdir(BASE_PATH):
            if file.endswith('.tar.gz'):
                tar_files.append(file)
            elif file.endswith('.csv'):
                print(f"  - {file} (metadata)")
        
        print(f"\nTar files found: {len(tar_files)}")
        for tf in tar_files:
            print(f"  - {tf}")