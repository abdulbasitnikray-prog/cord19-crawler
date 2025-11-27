
import json as js 
import os
import csv
import spacy 
import re

BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
EXTRACTION_FOLDER = os.path.join(BASE_PATH, "document_parses")

def get_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

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

def crawl_papers(csv_path, max_papers=None, output_file="crawled_papers.json"):  
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

    with open(output_file, 'w', encoding='utf-8') as f:
        js.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Crawled papers saved to {output_file}")
    return papers

if __name__ == "__main__":
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        exit(1)
    
    papers = crawl_papers(metadata_path, max_papers=1000, output_file="crawled_papers.json")