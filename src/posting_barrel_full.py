import json
import re
import os
import spacy
import tarfile
import csv
from collections import defaultdict
import math
import time

# Use your actual base path
BASE_PATH = "C:/Users/user/Downloads/cord-19_2020-04-10/2020-04-10"

# Text cleaning functions (from your crawler)
PUNCT_PATTERN = re.compile(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')))
SPACE_PATTERN = re.compile(r'\s+')
DIGIT_PATTERN = re.compile(r'\d+')

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

def get_scipacy_model():
    """Get spaCy model (from your crawler)"""
    disable_pipes = ["parser", "ner", "textcat", "custom"]
    try: 
        return spacy.load("en_core_sci_sm", disable=disable_pipes)
    except Exception as e:
        try:
            return spacy.load("en_core_web_sm", disable=disable_pipes)
        except:
            print(f"An exception occurred: {e}")
            return None

def stream_tar_dataset(metadata_path, tar_path, max_papers=None):
    """
    Stream papers from tar file (from your crawler, simplified)
    """
    print(f"Loading metadata from {metadata_path}...")
    meta_lookup = {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['sha']: 
                    for single_sha in row['sha'].split(';'):
                        meta_lookup[single_sha.strip()] = row
                if row['pmcid']:
                    meta_lookup[row['pmcid']] = row
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return []
    
    print(f"Streaming papers from {tar_path}...")
    found_count = 0
    
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if max_papers and found_count >= max_papers:
                    break
                
                if not member.isfile() or not member.name.endswith('.json'):
                    continue
                
                filename = os.path.basename(member.name).replace('.json', '').replace('.xml', '')
                
                if filename in meta_lookup:
                    meta_row = meta_lookup[filename]
                    f = tar.extractfile(member)
                    if f:
                        try:
                            content = json.load(f)
                            yield {
                                "cord_uid": meta_row["cord_uid"],
                                "title": meta_row["title"],
                                "json_parse": content
                            }
                            found_count += 1
                            if found_count % 1000 == 0:
                                print(f"  Streamed {found_count} papers...")
                        except:
                            pass
    except Exception as e:
        print(f"Error reading tar file: {e}")
    
    print(f"Total papers streamed: {found_count}")

def get_paper_stream(max_papers=None):
    """
    Get paper stream from all tar files
    """
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: Could not find metadata.csv at: {metadata_path}")
        return None
    
    # List of tar files (adjust based on what you have)
    tar_files = [
        "document_parses.tar.gz",  # Main file
        "biorxiv_medrxiv.tar.gz",
        "comm_use_subset.tar.gz", 
        "custom_license.tar.gz",
        "noncomm_use_subset.tar.gz"
    ]
    
    print(f"Creating paper stream from {BASE_PATH}")
    
    for tar_filename in tar_files:
        tar_path = os.path.join(BASE_PATH, tar_filename)
        
        if not os.path.exists(tar_path):
            print(f"Warning: Could not find {tar_filename}")
            continue
            
        print(f"\nProcessing {tar_filename}...")
        
        for paper in stream_tar_dataset(metadata_path, tar_path, max_papers):
            if max_papers:
                yield paper
            else:
                yield paper

def create_indexes_from_dataset(max_papers=1000, num_barrels=10):
    """
    Create posting lists and barrels from actual dataset
    """
    print("=" * 70)
    print(f"CREATING POSTING LISTS & BARRELS FROM CORD-19 DATASET")
    print(f"Base Path: {BASE_PATH}")
    print(f"Max Papers: {max_papers if max_papers else 'ALL'}")
    print(f"Number of Barrels: {num_barrels}")
    print("=" * 70)
    
    # Load spaCy model
    nlp = get_scipacy_model()
    if not nlp:
        print("ERROR: Could not load spaCy model")
        return
    
    # Initialize data structures
    lexicon = {}  # word -> word_id
    forward_index = {}  # doc_id -> [word_id1, word_id2, ...]
    inverted_index = defaultdict(dict)  # word_id -> {doc_id: frequency}
    backward_index = {}  # doc_id -> [lemmas]
    
    word_id_counter = 1
    processed_count = 0
    start_time = time.time()
    
    # Get paper stream
    paper_stream = get_paper_stream(max_papers)
    
    print("\n=== PROCESSING PAPERS ===")
    
    for paper in paper_stream:
        processed_data = process_papers(paper['json_parse'], nlp, paper['cord_uid'])
        
        if not processed_data or "tokens" not in processed_data:
            continue
            
        doc_id = paper["cord_uid"]
        tokens_list = processed_data["tokens"]
        doc_words_ids = []
        lemma_list = []
        
        # Process tokens for this paper
        for token in tokens_list:
            word = token["lemma"]
            lemma_list.append(word)
            
            # Build lexicon
            if word not in lexicon:
                lexicon[word] = word_id_counter
                word_id_counter += 1
            
            # Get word ID and update indexes
            w_id = lexicon[word]
            doc_words_ids.append(w_id)
            inverted_index[w_id][doc_id] = inverted_index[w_id].get(doc_id, 0) + 1
        
        # Store document indexes
        forward_index[doc_id] = doc_words_ids
        backward_index[doc_id] = lemma_list
        processed_count += 1
        
        # Progress reporting
        if processed_count % 100 == 0:
            elapsed_time = time.time() - start_time
            papers_per_second = processed_count / elapsed_time
            
            # Estimate remaining time
            if max_papers:
                remaining = max_papers - processed_count
                eta_seconds = remaining / papers_per_second if papers_per_second > 0 else 0
                eta_str = f"ETA: {eta_seconds/60:.1f} min"
            else:
                eta_str = ""
            
            print(f"  Processed {processed_count} papers | "
                  f"Speed: {papers_per_second:.1f} papers/sec | "
                  f"Unique words: {len(lexicon):,} | {eta_str}")
    
    total_time = time.time() - start_time
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Papers processed: {processed_count}")
    print(f"Unique words in lexicon: {len(lexicon):,}")
    print(f"Documents indexed: {len(forward_index)}")
    print(f"Terms in inverted index: {len(inverted_index):,}")
    
    return lexicon, forward_index, inverted_index, backward_index

def create_barrels(lexicon, num_barrels=10):
    """Partition lexicon into barrels for distributed processing"""
    print(f"\n=== PARTITIONING LEXICON INTO {num_barrels} BARRELS ===")
    
    # Sort words alphabetically
    sorted_words = sorted(lexicon.items(), key=lambda x: x[0])
    
    # Calculate words per barrel
    total_words = len(sorted_words)
    words_per_barrel = math.ceil(total_words / num_barrels)
    
    barrels = []
    current_barrel = {}
    barrel_id = 1
    
    print(f"Total words: {total_words:,}")
    print(f"Words per barrel: ~{words_per_barrel:,}")
    
    for i, (word, word_id) in enumerate(sorted_words):
        current_barrel[word] = word_id
        
        # Create new barrel when current is full or at end
        if len(current_barrel) >= words_per_barrel or i == total_words - 1:
            # Find actual range
            barrel_words = list(current_barrel.keys())
            range_start = barrel_words[0]
            range_end = barrel_words[-1]
            
            barrel_data = {
                "barrel_id": barrel_id,
                "range_start": range_start,
                "range_end": range_end,
                "word_count": len(current_barrel),
                "total_words": total_words,
                "lexicon": current_barrel.copy()
            }
            
            barrels.append(barrel_data)
            print(f"  Barrel {barrel_id}: Words '{range_start}' to '{range_end}' ({len(current_barrel):,} words)")
            
            current_barrel = {}
            barrel_id += 1
    
    return barrels

def create_posting_lists(inverted_index, lexicon):
    """Create detailed posting lists with statistics"""
    print(f"\n=== CREATING POSTING LISTS ===")
    
    posting_lists = {}
    
    for word_id, doc_freq in inverted_index.items():
        # Find word from lexicon
        word = [w for w, wid in lexicon.items() if wid == word_id][0]
        
        # Calculate statistics
        df = len(doc_freq)  # document frequency
        cf = sum(doc_freq.values())  # collection frequency
        
        posting_lists[word_id] = {
            "word_id": word_id,
            "word": word,
            "df": df,
            "cf": cf,
            "postings": doc_freq
        }
    
    # Print some statistics
    print(f"Created {len(posting_lists):,} posting lists")
    
    # Show top 10 most frequent terms
    sorted_postings = sorted(posting_lists.items(), 
                           key=lambda x: x[1]["cf"], 
                           reverse=True)[:10]
    
    print("\nTop 10 most frequent terms:")
    for word_id, data in sorted_postings:
        print(f"  '{data['word']}' (ID:{word_id}): DF={data['df']}, CF={data['cf']}")
    
    return posting_lists

def save_index_files(lexicon, forward_index, inverted_index, backward_index,
                    barrels, posting_lists, output_dir="cord19_indexes"):
    """Save all index files"""
    print(f"\n=== SAVING ALL FILES TO '{output_dir}' ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Lexicon
    lexicon_path = os.path.join(output_dir, "lexicon.json")
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, indent=None)
    print(f"✓ Saved lexicon.json ({len(lexicon):,} words)")
    
    # 2. Save Forward Index
    forward_path = os.path.join(output_dir, "forward_index.json")
    with open(forward_path, 'w', encoding='utf-8') as f:
        json.dump(forward_index, f, indent=None)
    print(f"✓ Saved forward_index.json ({len(forward_index):,} documents)")
    
    # 3. Save Inverted Index
    inverted_path = os.path.join(output_dir, "inverted_index.json")
    with open(inverted_path, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=None)
    print(f"✓ Saved inverted_index.json ({len(inverted_index):,} terms)")
    
    # 4. Save Backward Index
    backward_path = os.path.join(output_dir, "backward_index.json")
    with open(backward_path, 'w', encoding='utf-8') as f:
        json.dump(backward_index, f, indent=None)
    print(f"✓ Saved backward_index.json ({len(backward_index):,} documents)")
    
    # 5. Save Barrel Files
    barrels_dir = os.path.join(output_dir, "barrels")
    os.makedirs(barrels_dir, exist_ok=True)
    
    for barrel in barrels:
        filename = os.path.join(barrels_dir, f"barrel_{barrel['barrel_id']}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(barrel, f, indent=2)
    
    print(f"✓ Saved {len(barrels)} barrel files to {barrels_dir}/")
    
    # 6. Save Posting Lists (sampled for large datasets)
    postings_dir = os.path.join(output_dir, "posting_lists")
    os.makedirs(postings_dir, exist_ok=True)
    
    # Save only first 1000 posting lists as example (to avoid too many files)
    sample_count = min(1000, len(posting_lists))
    saved_count = 0
    
    for i, (word_id, data) in enumerate(posting_lists.items()):
        if i >= sample_count:
            break
            
        word = data["word"]
        safe_word = "".join(c for c in word if c.isalnum() or c == '_')
        filename = os.path.join(postings_dir, f"posting_{safe_word}_{word_id}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        saved_count += 1
    
    print(f"✓ Saved {saved_count} sample posting list files to {postings_dir}/")
    
    # 7. Save Statistics and Metadata
    stats = {
        "total_papers": len(forward_index),
        "total_unique_words": len(lexicon),
        "total_terms_in_index": len(inverted_index),
        "barrel_count": len(barrels),
        "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_path": BASE_PATH,
        "output_directory": output_dir,
        "lexicon_file": "lexicon.json",
        "forward_index_file": "forward_index.json",
        "inverted_index_file": "inverted_index.json",
        "backward_index_file": "backward_index.json",
        "barrels_directory": "barrels/",
        "posting_lists_directory": "posting_lists/",
        "barrel_ranges": [
            {
                "barrel_id": b["barrel_id"],
                "range_start": b["range_start"],
                "range_end": b["range_end"],
                "word_count": b["word_count"]
            } for b in barrels
        ]
    }
    
    stats_path = os.path.join(output_dir, "index_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved index_statistics.json")
    print(f"\n✅ ALL FILES SAVED SUCCESSFULLY!")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")

def main():
    """
    Main function to create posting lists and barrels from actual CORD-19 dataset
    """
    print("POSTING LIST & BARREL SYSTEM FOR CORD-19 DATASET")
    print("=" * 70)
    
    # Configuration
    MAX_PAPERS = 1000  # Set to None to process all papers, or a number for testing
    NUM_BARRELS = 10   # Number of barrels to partition lexicon into
    
    print(f"Configuration:")
    print(f"  - Max papers to process: {MAX_PAPERS if MAX_PAPERS else 'ALL'}")
    print(f"  - Number of barrels: {NUM_BARRELS}")
    print(f"  - Base path: {BASE_PATH}")
    print("=" * 70)
    
    # Check if base path exists
    if not os.path.exists(BASE_PATH):
        print(f"ERROR: Base path does not exist: {BASE_PATH}")
        print("Please update the BASE_PATH variable in the code.")
        return
    
    # Create indexes from actual dataset
    lexicon, forward_index, inverted_index, backward_index = create_indexes_from_dataset(
        max_papers=MAX_PAPERS, 
        num_barrels=NUM_BARRELS
    )
    
    if not lexicon:  # Check if indexing failed
        print("Indexing failed. Exiting.")
        return
    
    # Create posting lists
    posting_lists = create_posting_lists(inverted_index, lexicon)
    
    # Create barrels (partition lexicon)
    barrels = create_barrels(lexicon, num_barrels=NUM_BARRELS)
    
    # Save all files
    save_index_files(lexicon, forward_index, inverted_index, backward_index,
                    barrels, posting_lists, output_dir="cord19_index_output")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR BARREL INDEXING:")
    print("=" * 70)
    print("1. Each barrel (in barrels/ folder) contains a subset of the lexicon")
    print("2. Barrels can be processed in parallel on different machines")
    print("3. Posting lists are stored in inverted_index.json")
    print("4. Sample posting lists saved for testing")
    print("5. Use index_statistics.json for metadata")
    print("\nBarrel distribution:")
    for barrel in barrels:
        print(f"  Barrel {barrel['barrel_id']}: '{barrel['range_start']}' to '{barrel['range_end']}' ({barrel['word_count']:,} words)")
    print("=" * 70)

if __name__ == "__main__":
    main()