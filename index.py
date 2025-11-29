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
import time

from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#importing the crawler functions we need
from crawler import get_scipacy_model, stream_tar_dataset, process_papers, get_paper_stream, process_with_chunks

#creating the base paths for the directory and the folder in which we've stored the extracted .t.gaz folders
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
EXTRACTION_FOLDER = os.path.join(BASE_PATH, "document_parses")

#these two functions get our main models; the nlp model is for all our text preprocessing and the scispacy model is for our POS tagging for the lexicon
def get_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy model: python -m spacy download en_core_web_sm")
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

def generate_indexes_from_stream(paper_stream, max_papers=None):
    """
    Generate indexes directly from the paper stream without storing all processed papers in memory.
    sso more memory-efficient for large datasets like ours
    
    Returns:
        lexicon: {"word": word_id}
        forward_index: {"doc_id": [word_id1, word_id2, ...]}
        inverted_index: {"word_id": {"doc_id": frequency}}
    """
    lexicon = {}
    forward_index = {}
    inverted_index = {}
    word_id_counter = 1
    
    print("\n--- Generating Indexes from Stream ---")
    
    #only load scipacy once
    nlp = get_scipacy_model()
    if not nlp:
        print("ERROR: Could not load the spaCy model")
        return {}, {}, {}
    
    processed_count = 0
    start_time = time.time()
    
    for i, paper in enumerate(paper_stream):
        if max_papers and i >= max_papers:
            break
            
        #process paper
        processed_data = process_papers(paper['json_parse'], nlp, paper['cord_uid'])
        
        #skip papers that crashed during preprocessing
        if not processed_data or "tokens" not in processed_data:
            continue
            
        doc_id = paper["cord_uid"]
        tokens_list = processed_data["tokens"]
        doc_words_ids = []

        for token in tokens_list:
            word = token["lemma"]
            
            #this part constructs the lexicon
            if word not in lexicon:
                lexicon[word] = word_id_counter
                word_id_counter += 1
            
            #get the ID for the Word
            w_id = lexicon[word]
        
            #assign the id to the documents list
            doc_words_ids.append(w_id)
            
            #update Inverted Index
            if w_id not in inverted_index:
                inverted_index[w_id] = {}
            
            inverted_index[w_id][doc_id] = inverted_index[w_id].get(doc_id, 0) + 1
        
        #save to forward index
        forward_index[doc_id] = doc_words_ids
        processed_count += 1
        
        #progress reporting
        if processed_count % 100 == 0:
            elapsed_time = time.time() - start_time
            papers_per_second = processed_count / elapsed_time
            print(f"Processed {processed_count} papers | "
                  f"Speed: {papers_per_second:.1f} papers/sec | "
                  f"Unique words: {len(lexicon)}")
    
    total_time = time.time() - start_time
    print(f"Indexing completed in {total_time/60:.2f} minutes")
    print(f"Final stats: {processed_count} papers, {len(lexicon)} unique words")
    
    return lexicon, forward_index, inverted_index

def save_index_files(lexicon, forward_index, inverted_index, output_dir="."):
    """
    Save all index files to disk.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Lexicon
        lexicon_path = os.path.join(output_dir, "lexicon.json")
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            js.dump(lexicon, f, indent=None)
            print(f"SUCCESS: 'lexicon.json' created with {len(lexicon)} unique words.")

        # Save Forward Index
        forward_index_path = os.path.join(output_dir, "forward_index.json")
        with open(forward_index_path, 'w', encoding='utf-8') as f:
            js.dump(forward_index, f, indent=None)
            print(f"SUCCESS: 'forward_index.json' created with {len(forward_index)} documents.")
            
        # Save Inverted Index
        inverted_index_path = os.path.join(output_dir, "inverted_index.json")
        with open(inverted_index_path, 'w', encoding='utf-8') as f:
            js.dump(inverted_index, f, indent=None)
            print(f"SUCCESS: 'inverted_index.json' created with {len(inverted_index)} terms.")
            
    except Exception as e:
        print(f"Error saving files: {e}")

def main():
    print("--Starting Indexing--")
    
    #get paper stream and generate indexes directly
    paper_stream = get_paper_stream(max_papers=None)
    if not paper_stream:
        print("ERROR: Could not get paper stream")
        return
        
    lexicon, forward_index, inverted_index = generate_indexes_from_stream(paper_stream, max_papers=None)
    save_index_files(lexicon, forward_index, inverted_index)
    
    print("\n--Indexing Complete--")

if __name__ == "__main__":
    main()