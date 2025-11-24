import json as js 
import os
import csv
import spacy 
import nltk 
import string 
import re
import requests
import pandas as pd

from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

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
def local_metadatacsv_crawler(csv_path, max_papers=None):  
    papers = []
    found_count = 0

    with open(csv_path, 'r', encoding='utf-8') as within_f:
        reader = csv.DictReader(within_f)

        for i, row in enumerate(reader):

            if max_papers and len(papers) >= max_papers:
                break
                
            json_path = find_json_file(row)
            paper_text = None

            if json_path:
                try:
                    with open(json_path, "r", encoding="utf-8") as json_infile:
                        paper_text = js.load(json_infile)
                    found_count += 1
                except (js.JSONDecodeError, IOError) as e:
                    print(f"Error loading {json_path}: {e}")

            if paper_text:
                papers.append({
                    "cord_uid": row["cord_uid"], 
                    "title": row["title"], 
                    "abstract": row["abstract"], 
                    "json_parse": paper_text
                })

    print(f"Found {len(papers)} papers with JSON data")
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

def create_inverted_index(papers):
    inverted_index = {}

    for paper in papers:
        doc_id = paper["cord_uid"]

        if "processed" in paper and "tokens" in paper["processed"]:
            tokens = paper["processed"]["tokens"]
            term_freq ={}

            for token_data in tokens:
                term = token_data["lemma"]
                term_freq[term] =  term_freq.get(term, 0) + 1
            
            for term, freq in term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = {}
                inverted_index[term][doc_id] = freq

    return inverted_index
    
def extract_text(json_parse):
    if json_parse is None:
        return []
    
    body = json_parse.get("body_text", [])
    lines = []
    
    for section in body:
        text = section.get("text", "")
        lines.extend(text.splitlines())
        if len(lines) >= 3:  
            break

    return lines[:3]

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

def save_index_file(inverted_index, filename="inverted_index.json"):
    current_dir = os.getcwd()
    full_path = os.path.abspath(filename)

    print(f"Current directory: {current_dir}")
    print(f"Full file path: {full_path}")
    print(f"Inverted index size: {len(inverted_index)} terms")
    try:
        with open(filename, 'w', encoding='utf-8') as saved_file:
            js.dump(inverted_index, saved_file, indent=2, ensure_ascii=False) 
            #the ensure ascii arg is important because it allows text to be written in origianl representation

        if os.path.exists(filename):
            print(f"Inverted index saved to {full_path}")
            print(f"File exists: {os.path.exists(filename)}")
            print(f"File size: {os.path.getsize(filename)} bytes")
        else:
            print(f"File was not created at {full_path}")
            
    except Exception as e:
            print(f"Error occurred while saving the index: {e}")
    

def main():
    print("paths being checked")
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    print(f"Metadata path exists: {os.path.exists(metadata_path)}")
    print(f"Extraction folder exists: {os.path.exists(EXTRACTION_FOLDER)}")
    
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        return
    
    csv_path = os.path.join(BASE_PATH, "metadata.csv")
    papers = local_metadatacsv_crawler(csv_path, max_papers=10)  
    
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. needs installation.")
        nlp = None

    
    if not papers:
        print("No papers with JSON data found!")
        print("This could be because:")
        print("1. The JSON files don't exist in the expected locations")
        print("2. The SHA/PMCID values in metadata.csv don't match the file names")
        print("3. The document_parses folder structure is different than expected")
        return

    for i, paper in enumerate(papers):
        if paper["json_parse"] is not None:
            print(f"\n{'='*50}")
            print(f"Processing paper {i+1}: {paper['title'][:100]}...")
            processed_data = process_papers(paper['json_parse'], paper['cord_uid'])
            paper["processed"] = processed_data

    inverted_index = create_inverted_index(papers)
    save_index_file(inverted_index)

if __name__ == "__main__":
    main()
