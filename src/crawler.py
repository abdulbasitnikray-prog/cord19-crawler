import json as js 
import os
import csv
import spacy 
import re
import tarfile

# --- CONFIGURATION ---
BASE_PATH = "D:/Cord19/cord/2022"
# ---------------------

# Global Patterns
PUNCT_PATTERN = re.compile(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')))
SPACE_PATTERN = re.compile(r'\s+')
DIGIT_PATTERN = re.compile(r'\b\d+\b')

# Global variable for worker processes (Prevents reloading model 1000 times)
_nlp_model = None

def init_worker_nlp():
    """
    Called by multiprocessing workers to load the model ONCE.
    This fixes the ImportError in index.py
    """
    global _nlp_model
    if _nlp_model is None:
        disable_pipes = ["parser", "ner", "textcat", "custom", "attribute_ruler", "senter"]
        try:
            _nlp_model = spacy.load("en_core_sci_sm", disable=disable_pipes)
            _nlp_model.max_length = 2000000
        except Exception as e:
            try:
                _nlp_model = spacy.load("en_core_web_sm", disable=disable_pipes)
                _nlp_model.max_length = 2000000
            except:
                _nlp_model = None

def get_scipacy_model():
    """Helper for single-threaded usage"""
    disable_pipes = ["parser", "ner", "textcat", "custom", "attribute_ruler", "senter"]
    try: 
        nlp = spacy.load("en_core_sci_sm", disable=disable_pipes)
        nlp.max_length = 2000000 
        return nlp
    except Exception as e:
        try:
            return spacy.load("en_core_web_sm", disable=disable_pipes)
        except:
            return None

def clean_text(lines):
    if not lines: return ""
    full_text = " ".join(lines).lower()
    full_text = SPACE_PATTERN.sub(' ', full_text)
    full_text = PUNCT_PATTERN.sub('', full_text)
    full_text = DIGIT_PATTERN.sub('', full_text)
    return full_text.strip()

def extract_text(json_parse):
    if json_parse is None: return []
    body = json_parse.get("body_text", [])
    lines = []
    # 1. Get Title
    if 'metadata' in json_parse and 'title' in json_parse['metadata']:
        title = json_parse['metadata']['title']
        if title: lines.append(title)
    
    # 2. Get Abstract (Crucial for semantic training)
    if 'abstract' in json_parse:
        for entry in json_parse['abstract']:
            text = entry.get("text", "")
            if text: lines.append(text)
            
    # 3. Get Body Text
    if 'body_text' in json_parse:
        for section in json_parse['body_text']:
            text = section.get("text", "")
            if text:
                lines.append(text)
                if len(lines) >= 50: break 
                
    return lines

def stream_tar_dataset(metadata_path, tar_path, max_papers=None):
    """Streams papers from the .tar file matching metadata"""
    print("Step 1: Loading metadata...")
    meta_lookup = {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['sha']: 
                    for s in row['sha'].split(';'): meta_lookup[s.strip()] = row
                if row['pmcid']: meta_lookup[row['pmcid']] = row
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    print(f"Loaded metadata for {len(meta_lookup)} papers.")
    print(f"Step 2: Streaming from {tar_path}...")
    
    found_count = 0
    try:
        # 'r|*' allows reading both .tar and .tar.gz transparently
        with tarfile.open(tar_path, "r|*") as tar:
            for member in tar:
                if max_papers and found_count >= max_papers: break
                if not member.isfile() or not member.name.endswith('.json'): continue
                
                filename = os.path.basename(member.name).replace('.json', '').replace('.xml', '')
                
                if filename in meta_lookup:
                    meta_row = meta_lookup[filename]
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
                        except: pass
    except Exception as e:
        print(f"Error reading tar: {e}")

# --- COMPATIBILITY WRAPPERS (Required for index.py and train_semantic.py) ---
def get_paper_batches(batch_size=100, max_papers=None):
    """Bridge function to let other scripts iterate over papers"""
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    
    # Check for .tar OR .tar.gz automatically
    tar_name = "document_parses.tar"
    if not os.path.exists(os.path.join(BASE_PATH, tar_name)):
        tar_name = "document_parses.tar.gz"
        
    tar_path = os.path.join(BASE_PATH, tar_name)
    
    stream = stream_tar_dataset(metadata_path, tar_path, max_papers)
    
    current_batch = []
    for paper in stream:
        current_batch.append(paper)
        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch

def process_paper_batch(batch):
    """
    OPTIMIZED: Uses nlp.pipe() to process multiple papers at once.
    """
    # 1. Get the model
    global _nlp_model
    if _nlp_model:
        nlp = _nlp_model
    else:
        nlp = get_scipacy_model()
    
    if not nlp: return []

    # 2. Pre-extract text to prepare for batching
    # Creates a list of (uid, text) tuples
    docs_to_process = []
    for paper in batch:
        if paper.get('json_parse'):
            raw_lines = extract_text(paper['json_parse'])
            if raw_lines:
                text = clean_text(raw_lines)
                if text and len(text) > 50: # Skip tiny snippets
                    docs_to_process.append((paper['cord_uid'], text))
    
    results = []
    if not docs_to_process: return results

    # 3. Separate into lists for spaCy
    ids = [d[0] for d in docs_to_process]
    texts = [d[1] for d in docs_to_process]

    try:
        # n_process=1 because we are already inside a multiprocessing worker
        doc_stream = nlp.pipe(texts, batch_size=20, n_process=1)
        
        for doc, uid in zip(doc_stream, ids):
            indexed_tokens = []
            for token in doc:
                # Fast filtering
                if (token.is_stop or token.is_punct or token.is_space or 
                    token.like_num or len(token.text) < 2):
                    continue
                
                # Store data
                indexed_tokens.append({
                    'lemma': token.lemma_,
                    # 'pos': token.pos_,  # Optional: Comment out if you don't strictly need POS tags to save RAM
                    # 'tag': token.tag_
                })
            
            # Only add if we found valid tokens
            if indexed_tokens:
                results.append({
                    "cord_uid": uid,
                    "tokens": indexed_tokens
                })
                
    except Exception as e:
        print(f"Error in batch processing: {e}")

    return results
# -------------------------------------------------------------

def process_papers(json_parse, nlp):  
    if json_parse is None: return None
    raw_lines = extract_text(json_parse)
    if not raw_lines: return None
    full_text = clean_text(raw_lines)
    if not full_text: return None
    
    doc = nlp(full_text) 
    indexed_tokens = []
    
    for token in doc:
        if (token.is_stop or token.is_punct or token.is_space or 
            token.like_num or len(token.text) < 2):
            continue
        token_data = {'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_}
        indexed_tokens.append(token_data)
        
    return {"tokens": indexed_tokens}

def process_paper_single(json_parse, cord_uid=None):
    """
    Wrapper for processing a single paper on demand.
    Loads the model if needed and processes the JSON.
    """
    nlp = get_scipacy_model()
    if not nlp:
        return None
        
    result = process_papers(json_parse, nlp)
    
    # If a cord_uid was passed, ensure it's attached (optional depending on usage)
    if result and cord_uid:
        result['cord_uid'] = cord_uid
        
    return result

def save_files(lexicon, forward_index, inverted_index):
    # Ensure indexes directory exists
    os.makedirs("indexes", exist_ok=True)
    
    print(f"Saving files to /indexes folder...")
    with open("indexes/lexicon.json", 'w', encoding='utf-8') as f:
        js.dump(lexicon, f)
    with open("indexes/forward_index.json", 'w', encoding='utf-8') as f:
        js.dump(forward_index, f)
    with open("indexes/inverted_index.json", 'w', encoding='utf-8') as f:
        js.dump(inverted_index, f)
    print("Files saved successfully.")

def main():
    # This main block allows you to run "python crawler.py" to build indexes
    # But it also allows "import crawler" without running this part.
    print("--- Starting Single-Threaded Indexer ---")
    
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    
    # Auto-detect tar file
    tar_path = os.path.join(BASE_PATH, "document_parses.tar")
    if not os.path.exists(tar_path):
        tar_path = os.path.join(BASE_PATH, "document_parses.tar.gz")
        
    nlp = get_scipacy_model()
    
    lexicon = {}          
    forward_index = {}   
    inverted_index = {}  
    word_id_counter = 1   
    
    paper_stream = stream_tar_dataset(metadata_path, tar_path, max_papers=50000) 
    
    count = 0
    for paper in paper_stream:
        count += 1
        if count % 1000 == 0: print(f"Processing {count}...")
        
        processed = process_papers(paper['json_parse'], nlp)
        if not processed: continue
        
        doc_id = paper["cord_uid"]
        doc_word_ids = []
        
        for token in processed["tokens"]:
            word = token["lemma"]
            if word not in lexicon:
                lexicon[word] = word_id_counter
                word_id_counter += 1
            
            w_id = lexicon[word]
            doc_word_ids.append(w_id)
            
            if w_id not in inverted_index: inverted_index[w_id] = {}
            inverted_index[w_id][doc_id] = inverted_index[w_id].get(doc_id, 0) + 1
            
        forward_index[doc_id] = doc_word_ids
    
    save_files(lexicon, forward_index, inverted_index)

if __name__ == "__main__":
    main()