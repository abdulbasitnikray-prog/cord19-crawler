import json 
import os 
import math 
from collections import defaultdict, Counter
import time

def load_index_files(index_directory="indexes"):
    print(f"Loading index files from {index_directory}")
    
    indexes = {}
    files_to_load = {
        "lexicon": "lexicon.json",
        "forward_index": "forward_index.json", 
        "inverted_index": "inverted_index.json",
        "backward_index": "backward_index.json",
    }
    
    for name, f_name in files_to_load.items():
        filepath = os.path.join(index_directory, f_name)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    indexes[name] = json.load(f)
                print(f"Loaded {name} ({len(indexes[name]):,} entries)")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    return indexes

def save_barrels(barrels, doc_to_barrels, word_to_barrel, output_dir="barrels"):
    """Save barrel data to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each barrel
    for barrel in barrels:
        barrel_file = os.path.join(output_dir, f"barrel_{barrel['barrel_id']}.json")
        with open(barrel_file, 'w', encoding='utf-8') as f:
            json.dump(barrel, f, indent=2)
        print(f"Saved barrel {barrel['barrel_id']}")
    
    # Save mappings
    mappings = {
        "doc_to_barrels": doc_to_barrels,
        "word_to_barrel": word_to_barrel
    }
    
    mappings_file = os.path.join(output_dir, "barrel_mappings.json")
    with open(mappings_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"Saved barrel mappings")
    print(f"\nBarrels saved to: {os.path.abspath(output_dir)}")

def create_barrels_by_freq(inverted_index, lexicon, num_barrels=10):
    """Create barrels based on term frequency with even distribution"""
    
    # Create reverse mapping: word_id -> word
    id_to_word = {}
    for word, info in lexicon.items():
        if isinstance(info, dict) and "id" in info:
            id_to_word[str(info["id"])] = word
        else:
            # Handle case where lexicon is just word->id mapping
            id_to_word[str(info)] = word
    
    print(f"Creating {num_barrels} barrels from {len(inverted_index):,} terms...")
    
    word_frequencies = {}
    
    # Build word frequency statistics
    for word_id_str, doc_freq_dict in inverted_index.items():
        word = id_to_word.get(word_id_str)
        if word:
            doc_count = len(doc_freq_dict)
            total_freq = sum(doc_freq_dict.values())
            word_frequencies[word] = {
                "word_id": word_id_str,
                "doc_freq": doc_count,
                "total_freq": total_freq,
                "documents": list(doc_freq_dict.keys())
            }
    
    print(f"Collected frequencies for {len(word_frequencies):,} words")
    
    # Sort by document frequency (descending)
    sorted_words = sorted(word_frequencies.items(), 
                         key=lambda x: x[1]['doc_freq'], 
                         reverse=True)
    
    # Initialize barrels
    barrels = []
    for i in range(num_barrels):
        barrels.append({
            "barrel_id": i + 1,
            "words": {},
            "documents": set(),
            "word_count": 0,
            "total_doc_freq": 0,
            "total_word_freq": 0
        })
    
    # Distribute top frequent words evenly
    top_num_words = min(5000, len(sorted_words))
    
    for i, (word, stats) in enumerate(sorted_words[:top_num_words]):
        barrel_idx = i % num_barrels
        barrels[barrel_idx]["words"][word] = stats
        barrels[barrel_idx]["word_count"] += 1
        barrels[barrel_idx]["total_doc_freq"] += stats["doc_freq"]
        barrels[barrel_idx]["total_word_freq"] += stats["total_freq"]
    
    # Distribute remaining words to balance load
    for word, stats in sorted_words[top_num_words:]:
        # Find barrel with smallest total document frequency
        barrel_idx = min(range(num_barrels), 
                        key=lambda i: barrels[i]["total_doc_freq"])
        barrels[barrel_idx]["words"][word] = stats
        barrels[barrel_idx]["word_count"] += 1
        barrels[barrel_idx]["total_doc_freq"] += stats["doc_freq"]
        barrels[barrel_idx]["total_word_freq"] += stats["total_freq"]
    
    # Create word-to-barrel mapping
    word_to_barrel = {}
    for barrel in barrels:
        for word in barrel["words"]:
            word_to_barrel[word] = barrel["barrel_id"]
    
    # Create document-to-barrels mapping
    doc_to_barrel = defaultdict(set)
    
    for word, stats in word_frequencies.items():
        barrel_id = word_to_barrel.get(word)
        if barrel_id:
            for doc_id in stats["documents"]:
                doc_to_barrel[doc_id].add(barrel_id)
                
                # Add document to barrel's document set
                for barrel in barrels:
                    if barrel["barrel_id"] == barrel_id:
                        barrel["documents"].add(doc_id)
    
    # Convert sets to lists for JSON serialization
    for barrel in barrels:
        barrel["documents"] = sorted(list(barrel["documents"]))
        barrel["document_count"] = len(barrel["documents"])
    
    # Convert document-to-barrel mapping for serialization
    doc_to_barrels_serializable = {
        doc_id: sorted(list(barrel_ids))
        for doc_id, barrel_ids in doc_to_barrel.items()
    }
    
    # Add range information to barrels
    for barrel in barrels:
        if barrel["words"]:
            words_list = list(barrel["words"].keys())
            barrel["range_start"] = min(words_list)
            barrel["range_end"] = max(words_list)
        else:
            barrel["range_start"] = ""
            barrel["range_end"] = ""
    
    # Print statistics
    print(f"\nBarrel Statistics:")
    for barrel in barrels:
        print(f"  Barrel {barrel['barrel_id']}: {barrel['word_count']:,} words, "
              f"{barrel['document_count']:,} documents, "
              f"range: '{barrel['range_start'][:20]}...' to '{barrel['range_end'][:20]}...'")
    
    return barrels, doc_to_barrels_serializable, word_to_barrel

def create_and_save_barrels(index_directory="indexes", output_dir="barrels", num_barrels=10):
    """Create and save barrels - can be called from other modules"""
    print("=" * 70)
    print("BARREL CREATION FROM INDEXES")
    print("=" * 70)
    
    # Load indexes
    indexes = load_index_files(index_directory)
    
    if not indexes:
        print("Error: Could not load index files")
        return None, None, None
    
    # Create barrels
    barrels, doc_to_barrels, word_to_barrel = create_barrels_by_freq(
        indexes["inverted_index"],
        indexes["lexicon"],
        num_barrels=num_barrels
    )
    
    # Save barrels
    save_barrels(barrels, doc_to_barrels, word_to_barrel, output_dir)
    
    print("\n" + "=" * 70)
    print("BARREL CREATION COMPLETE")
    print("=" * 70)
    
    return barrels, doc_to_barrels, word_to_barrel

if __name__ == "__main__":
    create_and_save_barrels()