import tarfile
import os
import json


# Path to your massive dataset file
TAR_PATH = "D:/Cord19/cord/2022/document_parses.tar.gz"

# Where to save the small sample files
OUTPUT_DIR = "D:\cord19-crawler\sample_data"

# How many files you want
SAMPLE_SIZE = 50

def extract_samples():
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print(f"Opening {TAR_PATH} (Stream Mode)...")
    
    count = 0
    
    try:
        # "r:gz" opens it as a stream, so it doesn't load everything into RAM
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            for member in tar:
                # Stop if we have enough samples
                if count >= SAMPLE_SIZE:
                    break
                
                # We only want JSON files (skip folders and other junk)
                if not member.isfile() or not member.name.endswith('.json'):
                    continue
                
                # Extract the file object
                f = tar.extractfile(member)
                if f:
                    try:
                        # Read the JSON content
                        data = json.load(f)
                        
                        # Create a simple filename (just the ID)
                        # Original path might be "document_parses/pdf_json/abc.json"
                        # We want just "abc.json"
                        simple_filename = os.path.basename(member.name)
                        output_path = os.path.join(OUTPUT_DIR, simple_filename)
                        
                        # Save it to your sample_data folder
                        with open(output_path, 'w', encoding='utf-8') as outfile:
                            json.dump(data, outfile, indent=2)
                            
                        count += 1
                        print(f"[{count}/{SAMPLE_SIZE}] Extracted: {simple_filename}")
                        
                    except Exception as e:
                        print(f"Skipping broken file: {e}")

    except FileNotFoundError:
        print(f"Error: Could not find {TAR_PATH}")
        print("Please check the path in the script configuration.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n--- DONE ---")
    print(f"Successfully extracted {count} papers to the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    extract_samples()