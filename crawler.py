# crawler.py
import json as js 
import os
import csv
import tarfile
import time
import spacy 
import re

#creating the base paths for the directory and the folder in which we've stored the extracted .t.gaz folders
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
EXTRACTION_FOLDER = os.path.join(BASE_PATH, "document_parses")

# Global constraints preventing repetitive compile of regex patterns
PUNCT_PATTERN = re.compile(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')))
SPACE_PATTERN = re.compile(r'\s+')
DIGIT_PATTERN = re.compile(r'\d+')

def get_scipacy_model():
    disable_pipes =["parser","ner","textcat","custom"]
    try: 
        return spacy.load("en_core_sci_sm", disable=disable_pipes)
    except Exception as e:
        try:
            return spacy.load("en_core_web_sm", disable=disable_pipes)
        except:
            print(f"An exception occurred: {e}")
            return None

def clean_text(lines):
    """Clean text - only basic text processing"""
    if not lines: return ""
    full_text = " ".join(lines).lower()
    full_text = SPACE_PATTERN.sub(' ', full_text)
    full_text = PUNCT_PATTERN.sub('', full_text)
    full_text = DIGIT_PATTERN.sub('', full_text)
    return full_text.strip()

def extract_text(json_parse):
    """Extract text from paper JSON"""
    if json_parse is None:
        return []
    body = json_parse.get("body_text", [])
    lines = []
    for section in body:
        text = section.get("text", "")
        lines.extend(text.splitlines())
        if len(lines) >= 35:  
            break
    return lines[:35]

def process_papers(json_parse, nlp, cord_uid=None):  
    """Process single paper - text parsing only"""
    if json_parse is None:
        return None
    raw_lines = extract_text(json_parse)
    if not raw_lines: return None
    full_text = clean_text(raw_lines)
    if not full_text: return None
    nlp.max_length = 1500000 
    doc = nlp(full_text) 
    indexed_tokens = []
    for token in doc:
        if (token.is_stop or token.is_punct or token.is_space or 
            token.like_num or len(token.text) < 2):
            continue
        token_data = {
            'lemma': token.lemma_
        }
        indexed_tokens.append(token_data)
    return {"tokens": indexed_tokens}

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
                            if found_count % 100 == 0:
                                print(f"  Streamed {found_count} papers...")
                        except:
                            pass # Skip malformed JSONs
                            
    except FileNotFoundError:
        print(f"Error: Could not find the tar file at {tar_path}")
        return

    print(f"Successfully streamed {found_count} papers from archive.")

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
            
        print(f"Processing {tar_filename}...")
        
        # Get papers from this tar file
        for paper in stream_tar_dataset(metadata_path, tar_path, max_papers=None):
            if max_papers and total_found >= max_papers:
                return
            total_found += 1
            yield paper
            
            # Check again after yielding
            if max_papers and total_found >= max_papers:
                print(f"Reached max papers limit: {max_papers}")
                return
    
    print(f"Total papers available: {total_found}")

def process_with_chunks(paper_generator, chunk_size=100, max_papers=None):
    """
    Process papers in chunks directly from the yield generator.
    Returns processed papers with tokens - NO indexing!
    """
    all_processed_papers = []
    current_chunk = []
    total_processed = 0

    print("Processing papers in chunks...")
    start_time = time.time()
    
    # Load model once
    nlp = get_scipacy_model()
    if not nlp:
        print("ERROR: Could not load the spaCy model")
        return []
    
    # Process papers from generator in chunks
    for i, paper in enumerate(paper_generator):
        if max_papers and i >= max_papers:
            break

        current_chunk.append(paper)
        total_processed += 1

        # Process chunk when full
        if len(current_chunk) >= chunk_size:
            chunk_num = len(all_processed_papers) // chunk_size + 1
            print(f"Processing chunk {chunk_num} ({len(current_chunk)} papers)")

            processed_in_chunk = 0
            for paper in current_chunk:
                processed_data = process_papers(paper['json_parse'], nlp, paper['cord_uid'])
                if processed_data:
                    paper["processed"] = processed_data
                    all_processed_papers.append(paper)
                    processed_in_chunk += 1
            
            # Progress reporting
            elapsed_time = time.time() - start_time
            papers_per_second = total_processed / elapsed_time
            remaining_papers = max_papers - total_processed if max_papers else 0
            eta = remaining_papers / papers_per_second if papers_per_second > 0 else 0
            
            print(f"Progress: {total_processed}/{max_papers if max_papers else '∞'} | "
                  f"Speed: {papers_per_second:.1f} papers/sec | "
                  f"ETA: {eta/60:.1f} min")
            current_chunk = []
    
    # Process final chunk
    if current_chunk:
        print(f"Processing final chunk ({len(current_chunk)} papers)...")
        for paper in current_chunk:
            processed_data = process_papers(paper['json_parse'], nlp, paper['cord_uid'])
            if processed_data:
                paper["processed"] = processed_data
                all_processed_papers.append(paper)
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time/60:.2f} minutes")
    print(f"Successfully processed {len(all_processed_papers)} papers")
    
    return all_processed_papers

def check_files():
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    tar_path = os.path.join(BASE_PATH, "document_parses.tar.gz")
    
    print(f"Looking for metadata at: {metadata_path}")
    print(f"Exists: {os.path.exists(metadata_path)}")
    print(f"Looking for tar at: {tar_path}")
    print(f"Exists: {os.path.exists(tar_path)}")
    
    # List all files in BASE_PATH to see what's there
    if os.path.exists(BASE_PATH):
        print("\nFiles in directory:")
        for file in os.listdir(BASE_PATH):
            if file.endswith(('.tar', '.gz', '.csv')):
                print(f"  - {file}")