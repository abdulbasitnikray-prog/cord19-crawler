import json as js
import os
import csv
import pandas as pd
import spacy

#creating the base paths for the directory and the folder in which we've stored the extracted .t.gaz folders
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"
EXTRACTION_FOLDER = os.path.join(BASE_PATH, "document_parses")

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

def local_metadatacsv_crawler(csv_path, max_papers=None):  
    papers = []
    found_count = 0

    with open(csv_path, 'r', encoding='utf-8') as within_f:
        reader = csv.DictReader(within_f)

        for i, row in enumerate(reader):
            # Stop if we've reached max_papers
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

def extract_text(json_parse):
    if json_parse is None:
        return ""
   
    body = json_parse.get("body_text", [])
    lines = []
   
    for section in body:
        text = section.get("text", "")
        lines.extend(text.splitlines())
        if len(lines) >= 3:  
            break

    return lines[:3]

def process_papers(json_parse):  
    if json_parse is None:
        print("Warning: No JSON parse data available")
        return
   
    first_lines = extract_text(json_parse)

    for i, line in enumerate(first_lines):
        print(f"Line {i+1}: {line}")

def main():
    print("Checking paths...")
    metadata_path = os.path.join(BASE_PATH, "metadata.csv")
    print(f"Metadata path exists: {os.path.exists(metadata_path)}")
    print(f"Extraction folder exists: {os.path.exists(EXTRACTION_FOLDER)}")
   
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        return
   
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Please install it using:")
        print("python -m spacy download en_core_web_sm")
        nlp = None

    #
    csv_path = os.path.join(BASE_PATH, "metadata.csv")
    papers = local_metadatacsv_crawler(csv_path, max_papers=5)  
   
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
            print(f"CORD UID: {paper['cord_uid']}")
            process_papers(paper["json_parse"])  

if _name_ == "_main_":
    main()