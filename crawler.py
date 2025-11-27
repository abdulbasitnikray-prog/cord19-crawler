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
    
def get_scipacy_model():
    try: 
        return spacy.load("en_core_sci_sm")
    except Exception as e:
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
    
    papers = []
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
                            papers.append({
                                "cord_uid": meta_row["cord_uid"],
                                "title": meta_row["title"],
                                "json_parse": content
                            })
                            found_count += 1
                            if found_count % 100 == 0:
                                print(f"  Streamed {found_count} papers...")
                        except:
                            pass # Skip malformed JSONs
                            
    except FileNotFoundError:
        print(f"Error: Could not find the tar file at {tar_path}")
        return []

    print(f"Successfully streamed {len(papers)} papers from archive.")
    return papers

#word preprocessing functions from this point below 
def remove_stop_words(para_lines):
    nlp = get_nlp_model()
    if nlp is None:
        return para_lines
        
    cleaned_lines=[]

    for line in para_lines:
        if not line.strip():  
            continue

        doc = nlp(line)
        filtered_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_lines.append(" ".join(filtered_words))

    return cleaned_lines

def create_stem(cleaned_lines):
    stemmer = get_stemmer()
    if stemmer is None:
        return cleaned_lines
        
    stemmed_lines =[]

    for line in cleaned_lines:
        if not line.strip():
            stemmed_lines.append("")
            continue

        words = line.split()
        stemmed_words =[]

        for word in words:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)

        stemmed_lines.append(" ".join(stemmed_words))

    return stemmed_lines

def extract_named_entity(original_lines):  
    nlp = get_nlp_model()
    if nlp is None:
        return []
        
    entity_lines = []

    for line in original_lines:  
        if not line or not line.strip():
            continue
        doc = nlp(line)
        for ent in doc.ents:
            entity_lines.append((ent.text, ent.label_))  
    return entity_lines

def clean_text(text):
    cleaned = []

    for line in text:
        if not line or not line.strip():
            continue 

        line = line.lower()
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')), '', line)
        line = re.sub(r'\d+', '', line)
        if line.strip():  
                cleaned.append(line)
    return cleaned

def sentence_segments(text_lines):
    nlp = get_nlp_model()
    if nlp is None:
        return text_lines
        
    sentences =[]
    full_text = ' '.join(text_lines)

    doc = nlp(full_text)

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if len(sentence_text) > 7:
            sentences.append(sentence_text)
    
    return sentences

def pos_lemmatization(sentences):  
    nlp_model = get_scipacy_model()
    if nlp_model is None:
        return {"tokens": [], "total_tokens": 0, "content_words_count": 0}
    
    
    if isinstance(sentences, list):
        text = ' '.join(sentences)
    else:
        text = sentences
        
    doc = nlp_model(text)
    indexed_tokens =[]
    position = 0
    
    for token in doc:
        if (not token.is_alpha or token.is_stop or 
            token.is_punct or token.is_space):
            continue

        token_data = {
            'position': position,
            'original': token.text,
            'lemma': token.lemma_.lower(),
            'pos': token.pos_,  
            'pos_tag': token.tag_,
            'index_key': f"{token.lemma_.lower()}_{token.pos_}"
        }
        indexed_tokens.append(token_data)
        position += 1

    result = {
        'tokens': indexed_tokens,
        'total_tokens': len(indexed_tokens),
        'content_words_count': len([t for t in indexed_tokens if t['pos'] in ['NOUN', 'VERB', 'ADJ', 'ADV']])
    }
    return result

def extract_text(json_parse):
    if json_parse is None:
        return []
    
    body = json_parse.get("body_text", [])
    lines = []
    
    for section in body:
        text = section.get("text", "")
        lines.extend(text.splitlines())
        if len(lines) >= 100:  
            break

    return lines[:100]

def process_papers(json_parse, cord_uid):  
    if json_parse is None:
        print("Warning: No JSON parse data available")
        return
    
    unprocessed_lines = extract_text(json_parse)
    cleaned_lines = clean_text(unprocessed_lines)
    removed_stopword_lines = remove_stop_words(cleaned_lines)
    sentences = sentence_segments(removed_stopword_lines)
    lemmatized = pos_lemmatization(sentences)

    print(f"Processed {lemmatized['total_tokens']} tokens")
    print(f"Content words: {lemmatized['content_words_count']}")
    
    for i, token in enumerate(lemmatized['tokens'][:10]):
        print(f"   Token {i+1}: {token['original']} -> {token['lemma']} ({token['pos']})")
                
    return lemmatized

def generate_lexicon_and_data(papers):
    """
    Generates the Lexicon and saves the processed tokens for the Forward Index.
    """
    lexicon = {}          # Format: {"virus": 1, "cell": 2}
    processed_data = {}   # Format: {"doc_id": ["virus", "cell", "virus"]}
    word_id_counter = 1   # We start IDs at 1
    
    print("\n--- Generating Lexicon & Processed Data ---")

    for paper in papers:
        doc_id = paper["cord_uid"]
        
        # Skip papers that crashed during preprocessing
        if "processed" not in paper or "tokens" not in paper["processed"]:
            continue

        # Get the list of tokens (lemmatized words) from the paper
        tokens_list = paper["processed"]["tokens"]
        doc_words_list = []

        for token in tokens_list:
            word = token["lemma"]
            
            # 1. Build Lexicon: Assign a unique ID if the word hasnt been repeated
            if word not in lexicon:
                lexicon[word] = word_id_counter
                word_id_counter += 1
            
            # 2. Stores the word string for now
            doc_words_list.append(word)
        
        # Save this document's data
        processed_data[doc_id] = doc_words_list

    return lexicon, processed_data

def save_files(lexicon, processed_data):
    try:
        # Save Lexicon
        with open("lexicon.json", 'w', encoding='utf-8') as f:
            js.dump(lexicon, f, indent=None) # No indent to save space
            print(f"SUCCESS: 'lexicon.json' created with {len(lexicon)} unique words.")

        # Save Processed Data
        with open("processed_papers.json", 'w', encoding='utf-8') as f:
            js.dump(processed_data, f, indent=None)
            print(f"SUCCESS: 'processed_papers.json' created with {len(processed_data)} documents.")
            
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

    # 1. Crawl: Get papers
    print("Step 1: Loading papers...")
    # Extracting papers from the Stream_Tar method instead of the local_metadatacsv_crawler
    papers = stream_tar_dataset(metadata_path,tar_path, max_papers=50) 
    
    if not papers:
        print("No papers found.")
        return

    # 2. PreProcess: Clean the text
    print("Step 2: Preprocessing text ...")
    for i, paper in enumerate(papers):
        if paper["json_parse"] is not None:
            # Only print every 10th paper to keep terminal clean
            if i % 10 == 0: print(f"  Processing paper {i+1}/{len(papers)}...")
            
            processed_data = process_papers(paper['json_parse'], paper['cord_uid'])
            paper["processed"] = processed_data

    # 3. Lexicon: Create the Lexicon
    lexicon, processed_data = generate_lexicon_and_data(papers)
    
    # 4. Save: Saves the Files to Disk
    save_files(lexicon, processed_data)
    print("\n--- DONE ---")

if __name__ == "__main__":
    main()
