import pandas as pd
import json
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================

# 1. SETUP PATHS RELATIVE TO THIS SCRIPT
# Get the folder where THIS script (preprocess_data.py) is located (i.e., 'src')
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the Project Root (i.e., parent of 'src')
project_root = os.path.dirname(script_dir)

# Define the Output Path: Project Root -> data -> processed_corpus.csv
OUTPUT_FILE = os.path.join(project_root, 'data', 'processed_corpus.csv')


# 2. INPUT DATASET PATHS (Your external raw data)
DATASET_ROOT = r'D:/Cord19/cord/2022' 

METADATA_PATH = os.path.join(DATASET_ROOT, 'metadata.csv')
DOCUMENT_PARSES_PATH = os.path.join(DATASET_ROOT, 'document_parses')

# 3. Limit (Set to None for full run)
PROCESS_LIMIT = None

# ==========================================
# PROCESSING LOGIC (The rest stays the same...)

def get_file_content(json_relative_path, base_path):
    """
    Helper function to load a JSON file and extract body text.
    """
    if pd.isna(json_relative_path):
        return None
    
    # The metadata often separates multiple paths with '; '. We just take the first one.
    file_path = json_relative_path.split('; ')[0]
    full_path = os.path.join(base_path, file_path)

    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
                # Combine all paragraphs in the body_text
                text_content = " ".join([entry['text'] for entry in data['body_text']])
                return text_content
        except Exception as e:
            # If the file is corrupt or unreadable, ignore it
            return None
    return None

def process_dataset():
    print(f"Loading metadata from {METADATA_PATH}...")
    
    # Read the CSV (using low_memory=False to avoid mixed type warnings)
    df = pd.read_csv(METADATA_PATH, low_memory=False)
    
    # Apply limit if set (good for testing)
    if PROCESS_LIMIT:
        print(f"Test Mode: Processing only first {PROCESS_LIMIT} articles.")
        df = df.head(PROCESS_LIMIT).copy()
    else:
        print(f"Processing all {len(df)} articles. This may take a while...")

    # We will store results here
    final_data = []
    
    start_time = time.time()
    total_rows = len(df)

    print("Starting extraction...")

    for index, row in df.iterrows():
        # Print progress every 100 articles
        if index % 100 == 0:
            print(f"Processing row {index}/{total_rows}...", end='\r')

        title = row['title'] if pd.notna(row['title']) else "No Title"
        article_id = row['cord_uid']
        final_text = ""
        source_used = "None"

        # STRATEGY 1: Try PDF JSON
        if pd.notna(row['pdf_json_files']):
            content = get_file_content(row['pdf_json_files'], DOCUMENT_PARSES_PATH)
            if content:
                final_text = content
                source_used = "PDF JSON"

        # STRATEGY 2: If PDF failed, Try PMC JSON
        if not final_text and pd.notna(row['pmc_json_files']):
            content = get_file_content(row['pmc_json_files'], DOCUMENT_PARSES_PATH)
            if content:
                final_text = content
                source_used = "PMC JSON"

        # STRATEGY 3: If both JSONs failed, use Abstract
        if not final_text and pd.notna(row['abstract']):
            final_text = row['abstract']
            source_used = "Abstract (Fallback)"

        # Only add to our list if we found SOME text
        if final_text:
            final_data.append({
                'id': article_id,
                'title': title,
                'content': final_text,
                'source': source_used # useful for debugging
            })

    print(f"\nExtraction complete! Found content for {len(final_data)} articles.")
    
    # Save to new CSV
    print(f"Saving to {OUTPUT_FILE}...")
    out_df = pd.DataFrame(final_data)
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Done! Time taken: {round(time.time() - start_time, 2)} seconds.")

if __name__ == "__main__":
    process_dataset()