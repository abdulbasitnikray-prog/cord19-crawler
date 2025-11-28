import json as js 
import os
import csv
import spacy 
import nltk 
import string 
import re
import requests
import pandas as pd
import tarfile

from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#creating the base paths for the directory and the folder in which we've stored the extracted .t.gaz folders
BASE_PATH = "D:/Cord19/cord/2022"
EXTRACTION_FOLDER = os.path.join(BASE_PATH, "document_parses")

#these two functions get our main models; the nlp model is for all our text preprocessing and the scispacy model is for our POS tagging for the lexicon
def get_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

# Removed heavy components not being used for indexing
def get_scipacy_model():
    disable_pipes =["parser","ner","textcat","custom"]
    try: 
        return spacy.load("en_core_sci_sm",disable = disable_pipes)
    except Exception as e:
        try:
            return spacy.load("en_core_web_sm",disable = disable_pipes)
        except:
            print(f"An exception occurred: {e}")
            return None

def get_lemmatizer():
    try:
        return WordNetLemmatizer()
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None

def get_stemmer():
    try:
        return PorterStemmer()
    except Exception as e:
        print(f"An exception occurred: {e}")
        return None

#each paper will have a different directory depending on if it is in one of the three subfolders and then if it is a pdf or a pmc paper
#so we'll define the sub-folders and then we'll check the parameter given in the csv file if it has a "has_pdf_parse/has_pmc_xml_parse" or both
#depending on which one we have, we'll use the pmc_id(has_pmc_xml_parse) or sha(has_pdf_parse) and create the path using os.path.join(), checking to see if the path joined exists
def find_json_file(paper_row):
    if not os.path.exists(EXTRACTION_FOLDER):
        return None
    
    sub_folders = ["biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset", "custom_license"]

    if paper_row["has_pdf_parse"] == "True" and paper_row["sha"]:
        for folder in sub_folders:
            pdf_json_path = os.path.join(EXTRACTION_FOLDER, folder, "pdf_json", paper_row["sha"] + ".json")
            if os.path.exists(pdf_json_path):
                return pdf_json_path
    
    if paper_row["has_pmc_xml_parse"] == "True" and paper_row["pmcid"]:
        for folder in sub_folders:
            pmc_json_path = os.path.join(EXTRACTION_FOLDER, folder, "pmc_json", paper_row["pmcid"] + ".xml.json")
            if os.path.exists(pmc_json_path):
                return pmc_json_path
    
    return None

#for a small test release, the crawler has a max_papers argument so it will crawl the first nth papers that we ask it to crawl
#through. if the path is correct for the folder, it will create a dictionary with cord_id, title, abstract and json_parse
#json parse is the paper text.

#Replaced Already Present "local_metadatacsv_crawler" with a stream_tar parser
def stream_tar_dataset(metadata_path, tar_path, max_papers=None):
    """
    Reads directly from the .tar file without extracting it.
    Matches files in the TAR to entries in metadata.csv.
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
        return []

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
                # Example: document_parses/pdf_json/000a0fcbf.json -> 000a0fcbf
                filename = os.path.basename(member.name).replace('.json', '').replace('.xml', '')
                
                # 3. Check if this file exists in our metadata
                if filename in meta_lookup:
                    meta_row = meta_lookup[filename]
                    
                    # Read the JSON file directly from the TAR stream
                    f = tar.extractfile(member)
                    if f:
                        try:
                            content = js.load(f)
                            yield{
                                "cord_uid": meta_row["cord_uid"],
                                "title": meta_row["title"],
                                "json_parse": content
                            }
                            found_count += 1
                        except:
                            pass # Skip malformed JSONs
                            
    except FileNotFoundError:
        print(f"Error: Could not find the tar file at {tar_path}")
        return []

    print(f"Successfully streamed {found_count} papers from archive.")

#Global Constraints preventing repetitive compile of regex patterns

PUNCT_PATTERN = re.compile(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')))
SPACE_PATTERN = re.compile(r'\s+')
DIGIT_PATTERN = re.compile(r'\b\d+\b')

def clean_text(lines):
    """
    Joins lines into one string and cleans it using pre-compiled regex.
    Returns: Single String
    """
    if not lines: return ""
    
    # 1. Join into one block
    full_text = " ".join(lines).lower()
    
    # 2. Apply Regex on the whole block (Very Fast)
    full_text = SPACE_PATTERN.sub(' ', full_text)
    full_text = PUNCT_PATTERN.sub('', full_text)
    full_text = DIGIT_PATTERN.sub('', full_text)
    
    return full_text.strip()

def extract_text(json_parse):
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

def process_papers(json_parse, nlp):  
    if json_parse is None:
        print("Warning: No JSON parse data available")
        return
    
    # 1. Get raw text lines
    raw_lines = extract_text(json_parse)
    if not raw_lines: return None
    
    # 2. Clean and Join (New Helper)
    full_text = clean_text(raw_lines)
    if not full_text: return None
    
    # 3. Increase limit just in case (though 35 lines won't hit it)
    nlp.max_length = 1500000 
    
    # 4. Run the Model ONCE (Tokenize + Tag + Lemmatize)
    doc = nlp(full_text) 
    
    indexed_tokens = []
    
    # 5. Fast Iteration
    for token in doc:
        # Combined Filter: Stop words, Punctuation, Numbers, Short words
        if (token.is_stop or token.is_punct or token.is_space or 
            token.like_num or len(token.text) < 2):
            continue

        # We only store the lemma. 
        # Position is implicit by the order in the list if needed later.
        token_data = {
            'lemma': token.lemma_
        }
        indexed_tokens.append(token_data)

    return {"tokens": indexed_tokens}

# Merged the following method within the main for pipeline implementation
"""
def generate_lexicon_and_forward_index(papers):
    
    Generates the Lexicon and saves the processed tokens for the Forward Index.

    lexicon = {}          # Format: {"virus": 1, "cell": 2}
    forward_index = {}   # Format: {"doc_id": [1, 2, 3]}
    word_id_counter = 1   # We start IDs at 1
    
    print("\n--- Generating Lexicon & Processed Data ---")

    for paper in papers:
        doc_id = paper["cord_uid"]
        
        # Skip papers that crashed during preprocessing
        if "processed" not in paper or "tokens" not in paper["processed"]:
            continue

        # Get the list of tokens (lemmatized words) from the paper
        tokens_list = paper["processed"]["tokens"]
        doc_words_ids = []

        for token in tokens_list:
            word = token["lemma"]
            
            # 1. Build Lexicon: Assign a unique ID if the word hasnt been repeated
            if word not in lexicon:
                lexicon[word] = word_id_counter
                word_id_counter += 1
            
        # Get the ID for the Word
            w_id = lexicon[word]
        
        # Assign the id to the documents list
            doc_words_ids.append(w_id)
        
        # Save this document's data
        forward_index[doc_id] = doc_words_ids

    return lexicon, forward_index
"""
def save_files(lexicon, forward_index,inverted_index):
    try:
        # Save Lexicon
        with open("lexicon.json", 'w', encoding='utf-8') as f:
            js.dump(lexicon, f, indent=None) # No indent to save space
            print(f"SUCCESS: 'lexicon.json' created with {len(lexicon)} unique words.")

        # Save Forward Index
        with open("forward_index.json", 'w', encoding='utf-8') as f:
            js.dump(forward_index, f, indent=None)
            print(f"SUCCESS: 'forward_index.json' created with {len(forward_index)} documents.")
        # Save Inverted Index
        with open("inverted_index.json", 'w', encoding='utf-8') as f:
            js.dump(inverted_index, f, indent=None)
            print(f"SUCCESS: 'inverted_index.json' ({len(inverted_index)} terms)")
    except Exception as e:
        print(f"Error saving files: {e}")

def main():
    print("--- Starting Project Pipeline ---")
    
    # Check if paths exist
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")

    # Checks for the TAR File
    tar_path = os.path.join(BASE_PATH, "document_parses.tar.gz")
    if not os.path.exists(metadata_path):
        print(f"ERROR: Could not find metadata.csv at: {metadata_path}")
        print("ACTION: Please edit the 'BASE_PATH' variable in the code.")
        return
    # 0. Load Model
    nlp = get_scipacy_model()
    if not nlp: return

    # Merged the Previous Lexicon & ForwardIndex Function within the main
    # Lexi_FI: Initializition of Dicts
    lexicon = {}          # Format: {"virus": 1, "cell": 2}
    forward_index = {}   # Format: {"doc_id": [1, 2, 3]}
    inverted_index = {}  # Format {"1": [doc1,doc2,doc3]}
    word_id_counter = 1   # We start IDs at 1
    
    # 1. Crawl: Get papers
    print("Step 1: Loading papers as a stream...")
    # Extracting papers from the Stream_Tar method instead of the local_metadatacsv_crawler
    paper_stream = stream_tar_dataset(metadata_path,tar_path, max_papers=100) 
    
    if not paper_stream:
        print("No papers found.")
        return

    # 2. PreProcess: Clean the text
    print("Step 2: Preprocessing text ...")
    print("\n--- Generating Lexicon & Forward Index ---")
    for i, paper in enumerate(paper_stream):
        if paper["json_parse"] is not None:
            #Print every paper 
            if i % 1000 == 0: print(f"  Processing paper {i+1} - {i+999} ...")
            processed_data = process_papers(paper['json_parse'], nlp)


        # Skip papers that crashed during preprocessing
        if not processed_data or "tokens" not in processed_data:
            continue

        # Get the list of tokens (lemmatized words) from the paper
        doc_id = paper["cord_uid"]
        tokens_list = processed_data["tokens"]
        doc_words_ids = []

        for token in tokens_list:
            word = token["lemma"]
            
            # 1. Build Lexicon: Assign a unique ID if the word hasnt been repeated
            if word not in lexicon:
                lexicon[word] = word_id_counter
                word_id_counter += 1
            
            # Get the ID for the Word
            w_id = lexicon[word]
        
            # Assign the id to the documents list
            doc_words_ids.append(w_id)
            # Update Inverted Index (Map Word ID -> Doc ID)
            if w_id not in inverted_index:
                inverted_index[w_id] = {}
            
            # Increment frequency: {doc_id: count}
            # Using .get() for safe incrementing
            inverted_index[w_id][doc_id] = inverted_index[w_id].get(doc_id, 0) + 1
        
        # Save this document's data
        forward_index[doc_id] = doc_words_ids
    
    # 4. Save: Saves the Files to Disk
    save_files(lexicon, forward_index,inverted_index)
    print("\n--- DONE ---")

if __name__ == "__main__":
    main()
