import json
import re
import os
import spacy
from collections import defaultdict
import math

# Sample paper data (embedded in the code)
SAMPLE_PAPER = {
    "paper_id": "025339bfce1cb8efa81c5accdabefe04dcdac9d2",
    "metadata": {
        "title": "Managing emerging infectious diseases: Is a federal system an impediment to effective laws?",
        "authors": [
            {
                "first": "Genevieve",
                "middle": [],
                "last": "Howse",
                "suffix": "",
                "affiliation": {
                    "laboratory": "",
                    "institution": "La Trobe University",
                    "location": {
                        "settlement": "Vic",
                        "country": "Australia"
                    }
                },
                "email": "g.howse@latrobe.edu.au"
            }
        ]
    },
    "abstract": [
        {
            "text": "In the 1980's and 1990's HIV/AIDS was the emerging infectious disease. In 2003-2004 we saw the emergence of SARS, Avian influenza and Anthrax in a man made form used for bioterrorism. Emergency powers legislation in Australia is a patchwork of Commonwealth quarantine laws and State and Territory based emergency powers in public health legislation. It is time for a review of such legislation and time for consideration of the efficacy of such legislation from a country wide perspective in an age when we have to consider the possibility of mass outbreaks of communicable diseases which ignore jurisdictional boundaries.",
            "cite_spans": [],
            "ref_spans": [],
            "section": "Abstract"
        }
    ],
    "body_text": [
        {
            "text": "The management of infectious diseases in an increasingly complex world of mass international travel, globalization and terrorism heightens challenges for Federal, State and Territory Governments in ensuring that Australia's laws are sufficiently flexible to address the types of problems that may emerge.",
            "cite_spans": [],
            "ref_spans": [],
            "section": ""
        },
        {
            "text": "In the 1980's and 1990's HIV/AIDS was the latest \"emerging infectious disease\". Considerable thought was put into the legislative response by a number of Australian jurisdictions. Particular attention had to be given to the unique features of the disease such as the method of transmission, the kinds of people who were at risk, and the protections needed by the community and the infected population to best manage the care of those infected and to minimize new infections.",
            "cite_spans": [],
            "ref_spans": [],
            "section": ""
        }
    ]
}

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
    
    # Combine abstract and body text
    lines = []
    
    # Add abstract
    for abstract_section in json_parse.get("abstract", []):
        lines.extend(abstract_section.get("text", "").splitlines())
    
    # Add body text (limited for demo)
    body = json_parse.get("body_text", [])
    for i, section in enumerate(body[:3]):  # First 3 paragraphs only
        text = section.get("text", "")
        lines.extend(text.splitlines())
    
    return lines

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

def create_indexes_from_single_paper():
    """Create all indexes from the single sample paper"""
    print("=== Creating Indexes from Single Paper ===")
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Initialize data structures
    lexicon = {}  # word -> word_id
    forward_index = {}  # doc_id -> [word_id1, word_id2, ...]
    inverted_index = defaultdict(dict)  # word_id -> {doc_id: frequency}
    backward_index = {}  # doc_id -> [lemmas]
    positional_index = defaultdict(lambda: defaultdict(list))  # word_id -> {doc_id: [positions]}
    
    doc_id = "paper_001"
    processed_data = process_papers(SAMPLE_PAPER, nlp, doc_id)
    
    if not processed_data or "tokens" not in processed_data:
        print("No tokens extracted!")
        return
    
    tokens_list = processed_data["tokens"]
    doc_words_ids = []
    lemma_list = []
    word_id_counter = 1
    
    print(f"\nProcessing paper: {SAMPLE_PAPER['metadata']['title']}")
    print(f"Extracted {len(tokens_list)} tokens")
    
    # Process each token
    position = 0
    for token in tokens_list:
        word = token["lemma"]
        lemma_list.append(word)
        
        # Build lexicon (word -> word_id)
        if word not in lexicon:
            lexicon[word] = word_id_counter
            word_id_counter += 1
        
        # Get word ID
        w_id = lexicon[word]
        doc_words_ids.append(w_id)
        
        # Update inverted index (posting list with frequencies)
        inverted_index[w_id][doc_id] = inverted_index[w_id].get(doc_id, 0) + 1
        
        # Update positional index (word -> positions)
        positional_index[word][doc_id].append(position)
        position += 1
    
    # Store in indexes
    forward_index[doc_id] = doc_words_ids
    backward_index[doc_id] = lemma_list
    
    print(f"\n=== INDEXES CREATED ===")
    print(f"Lexicon size: {len(lexicon)} unique words")
    print(f"Forward index: {len(forward_index)} document(s)")
    print(f"Inverted index: {len(inverted_index)} terms with posting lists")
    print(f"Positional index: {len(positional_index)} terms with positions")
    
    return lexicon, forward_index, inverted_index, backward_index, positional_index

def create_barrels(lexicon, num_barrels=3):
    """Partition lexicon into barrels (alphabetically)"""
    print(f"\n=== PARTITIONING LEXICON INTO {num_barrels} BARRELS ===")
    
    # Sort words alphabetically
    sorted_words = sorted(lexicon.items(), key=lambda x: x[0])
    
    # Calculate words per barrel
    total_words = len(sorted_words)
    words_per_barrel = math.ceil(total_words / num_barrels)
    
    barrels = []
    current_barrel = {}
    barrel_id = 1
    
    for i, (word, word_id) in enumerate(sorted_words):
        current_barrel[word] = word_id
        
        # Create new barrel when current is full or at end
        if len(current_barrel) >= words_per_barrel or i == total_words - 1:
            # Find actual range (first and last words in barrel)
            barrel_words = list(current_barrel.keys())
            range_start = barrel_words[0]
            range_end = barrel_words[-1]
            
            barrel_data = {
                "barrel_id": barrel_id,
                "range_start": range_start,
                "range_end": range_end,
                "word_count": len(current_barrel),
                "lexicon": current_barrel.copy()
            }
            
            barrels.append(barrel_data)
            print(f"Barrel {barrel_id}: Words '{range_start}' to '{range_end}' ({len(current_barrel)} words)")
            
            current_barrel = {}
            barrel_id += 1
    
    return barrels

def create_posting_lists(inverted_index, lexicon):
    """Create detailed posting lists"""
    print(f"\n=== CREATING POSTING LISTS ===")
    
    posting_lists = {}
    
    for word_id, doc_freq in inverted_index.items():
        # Find word from lexicon
        word = [w for w, wid in lexicon.items() if wid == word_id][0]
        
        # Calculate document frequency (df) and collection frequency (cf)
        df = len(doc_freq)  # number of documents containing this word
        cf = sum(doc_freq.values())  # total occurrences across all documents
        
        posting_lists[word_id] = {
            "word_id": word_id,
            "word": word,
            "df": df,
            "cf": cf,
            "postings": doc_freq  # {doc_id: frequency}
        }
        
        print(f"Word '{word}' (ID:{word_id}): DF={df}, CF={cf}")
    
    return posting_lists

def save_all_files(lexicon, forward_index, inverted_index, backward_index, 
                   positional_index, barrels, posting_lists, output_dir="index_output"):
    """Save all index files"""
    print(f"\n=== SAVING ALL FILES TO '{output_dir}' ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Lexicon
    with open(f"{output_dir}/lexicon.json", "w") as f:
        json.dump(lexicon, f, indent=2)
    print(f"✓ Saved lexicon.json ({len(lexicon)} words)")
    
    # 2. Save Forward Index
    with open(f"{output_dir}/forward_index.json", "w") as f:
        json.dump(forward_index, f, indent=2)
    print(f"✓ Saved forward_index.json ({len(forward_index)} documents)")
    
    # 3. Save Inverted Index
    with open(f"{output_dir}/inverted_index.json", "w") as f:
        json.dump(inverted_index, f, indent=2)
    print(f"✓ Saved inverted_index.json ({len(inverted_index)} terms)")
    
    # 4. Save Backward Index
    with open(f"{output_dir}/backward_index.json", "w") as f:
        json.dump(backward_index, f, indent=2)
    print(f"✓ Saved backward_index.json ({len(backward_index)} documents)")
    
    # 5. Save Positional Index
    with open(f"{output_dir}/positional_index.json", "w") as f:
        json.dump(positional_index, f, indent=2)
    print(f"✓ Saved positional_index.json ({len(positional_index)} terms)")
    
    # 6. Save Barrel Files
    barrels_dir = f"{output_dir}/barrels"
    os.makedirs(barrels_dir, exist_ok=True)
    
    for barrel in barrels:
        filename = f"{barrels_dir}/barrel_{barrel['barrel_id']}.json"
        with open(filename, "w") as f:
            json.dump(barrel, f, indent=2)
    print(f"✓ Saved {len(barrels)} barrel files to {barrels_dir}/")
    
    # 7. Save Posting Lists (individual files for each term)
    postings_dir = f"{output_dir}/posting_lists"
    os.makedirs(postings_dir, exist_ok=True)
    
    for word_id, posting_data in posting_lists.items():
        word = posting_data["word"]
        safe_word = "".join(c for c in word if c.isalnum() or c == '_')
        filename = f"{postings_dir}/posting_{safe_word}_{word_id}.json"
        with open(filename, "w") as f:
            json.dump(posting_data, f, indent=2)
    print(f"✓ Saved {len(posting_lists)} posting list files to {postings_dir}/")
    
    # 8. Save Master Index File (summary)
    master_index = {
        "total_words": len(lexicon),
        "total_documents": len(forward_index),
        "barrel_count": len(barrels),
        "lexicon_file": "lexicon.json",
        "forward_index_file": "forward_index.json",
        "inverted_index_file": "inverted_index.json",
        "positional_index_file": "positional_index.json",
        "barrels_directory": "barrels/",
        "posting_lists_directory": "posting_lists/",
        "sample_query_words": list(lexicon.keys())[:10]  # First 10 words for testing
    }
    
    with open(f"{output_dir}/MASTER_INDEX.json", "w") as f:
        json.dump(master_index, f, indent=2)
    print(f"✓ Saved MASTER_INDEX.json (summary file)")
    
    print(f"\n✅ ALL FILES SAVED SUCCESSFULLY!")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")

def print_sample_data(lexicon, inverted_index, barrels, posting_lists):
    """Print sample data for verification"""
    print(f"\n=== SAMPLE DATA ===")
    
    print("\n1. LEXICON (first 10 words):")
    for i, (word, word_id) in enumerate(list(lexicon.items())[:10]):
        print(f"   '{word}' -> ID:{word_id}")
    
    print("\n2. POSTING LISTS (first 5 terms):")
    for i, (word_id, data) in enumerate(list(posting_lists.items())[:5]):
        word = data["word"]
        df = data["df"]
        cf = data["cf"]
        print(f"   '{word}' (ID:{word_id}): In {df} document(s), {cf} total occurrences")
    
    print("\n3. BARRELS:")
    for barrel in barrels:
        words = list(barrel["lexicon"].keys())
        print(f"   Barrel {barrel['barrel_id']}: '{barrel['range_start']}' to '{barrel['range_end']}'")
        print(f"      Contains: {words[:3]}..." if len(words) > 3 else f"      Contains: {words}")
    
    print("\n4. QUERY EXAMPLE:")
    if lexicon:
        sample_word = list(lexicon.keys())[5] if len(lexicon) > 5 else list(lexicon.keys())[0]
        word_id = lexicon[sample_word]
        if word_id in inverted_index:
            posting = inverted_index[word_id]
            print(f"   Query for '{sample_word}':")
            for doc_id, freq in posting.items():
                print(f"      Document '{doc_id}': appears {freq} time(s)")

def main():
    """Main function to create all indexes from single paper"""
    print("=" * 60)
    print("POSTING LIST & BARREL PARTITIONING DEMONSTRATION")
    print("Using single CORD-19 paper sample")
    print("=" * 60)
    
    # Create all indexes
    lexicon, forward_index, inverted_index, backward_index, positional_index = create_indexes_from_single_paper()
    
    # Create posting lists
    posting_lists = create_posting_lists(inverted_index, lexicon)
    
    # Create barrels (partition lexicon)
    barrels = create_barrels(lexicon, num_barrels=3)
    
    # Print sample data
    print_sample_data(lexicon, inverted_index, barrels, posting_lists)
    
    # Save all files
    save_all_files(lexicon, forward_index, inverted_index, backward_index,
                  positional_index, barrels, posting_lists)
    
    print(f"\n" + "=" * 60)
    print("NEXT STEPS FOR ARYAN:")
    print("1. Use barrel files for distributed indexing")
    print("2. Use posting lists for query processing")
    print("3. Scale this structure to full dataset")
    print("=" * 60)

if __name__ == "__main__":
    main()