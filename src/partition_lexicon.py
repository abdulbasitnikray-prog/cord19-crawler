"""
partition_lexicon.py

Functions for partitioning lexicon into barrels for distributed processing.

Barrels divide the lexicon into manageable chunks that can be processed
in parallel across multiple machines or threads.
"""

import json
import os
import math
from typing import Dict, List, Tuple, Any
import time

def load_lexicon(lexicon_file: str) -> Dict:
    """
    Load lexicon from JSON file.
    
    Args:
        lexicon_file: Path to lexicon JSON file
        
    Returns:
        Dictionary mapping words to word IDs
    """
    try:
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            lexicon = json.load(f)
        print(f"Loaded lexicon with {len(lexicon):,} words from {lexicon_file}")
        return lexicon
    except Exception as e:
        print(f"Error loading lexicon: {e}")
        return {}

def create_alphabetical_barrels(lexicon: Dict, num_barrels: int = 10) -> List[Dict]:
    """
    Partition lexicon into barrels alphabetically.
    
    Args:
        lexicon: Dictionary of word -> word_id mappings
        num_barrels: Number of barrels to create
        
    Returns:
        List of barrel dictionaries
    """
    if not lexicon:
        print("Empty lexicon, cannot create barrels")
        return []
    
    # Sort words alphabetically
    sorted_words = sorted(lexicon.items(), key=lambda x: x[0])
    
    # Calculate optimal barrel size
    total_words = len(sorted_words)
    words_per_barrel = math.ceil(total_words / num_barrels)
    
    barrels = []
    current_barrel = {}
    barrel_id = 1
    
    print(f"Partitioning {total_words:,} words into {num_barrels} barrels")
    print(f"Target: ~{words_per_barrel:,} words per barrel")
    
    for i, (word, word_id) in enumerate(sorted_words):
        current_barrel[word] = word_id
        
        # Create new barrel when current is full or at end
        if len(current_barrel) >= words_per_barrel or i == total_words - 1:
            barrel_words = list(current_barrel.keys())
            range_start = barrel_words[0]
            range_end = barrel_words[-1]
            
            barrel_data = {
                "barrel_id": barrel_id,
                "range_start": range_start,
                "range_end": range_end,
                "word_count": len(current_barrel),
                "total_words": total_words,
                "lexicon": current_barrel.copy()
            }
            
            barrels.append(barrel_data)
            print(f"  Barrel {barrel_id}: '{range_start}' to '{range_end}' ({len(current_barrel):,} words)")
            
            current_barrel = {}
            barrel_id += 1
    
    return barrels

def create_size_based_barrels(lexicon: Dict, max_barrel_size: int = 10000) -> List[Dict]:
    """
    Partition lexicon into barrels based on maximum size.
    
    Args:
        lexicon: Dictionary of word -> word_id mappings
        max_barrel_size: Maximum words per barrel
        
    Returns:
        List of barrel dictionaries
    """
    if not lexicon:
        return []
    
    # Sort words alphabetically
    sorted_words = sorted(lexicon.items(), key=lambda x: x[0])
    
    barrels = []
    current_barrel = {}
    barrel_id = 1
    
    print(f"Partitioning lexicon into barrels of max {max_barrel_size:,} words")
    
    for i, (word, word_id) in enumerate(sorted_words):
        current_barrel[word] = word_id
        
        # Create new barrel when current reaches max size or at end
        if len(current_barrel) >= max_barrel_size or i == len(sorted_words) - 1:
            barrel_words = list(current_barrel.keys())
            
            barrel_data = {
                "barrel_id": barrel_id,
                "range_start": barrel_words[0],
                "range_end": barrel_words[-1],
                "word_count": len(current_barrel),
                "max_size": max_barrel_size,
                "lexicon": current_barrel.copy()
            }
            
            barrels.append(barrel_data)
            print(f"  Barrel {barrel_id}: {len(current_barrel):,} words "
                  f"('{barrel_words[0]}' to '{barrel_words[-1]}')")
            
            current_barrel = {}
            barrel_id += 1
    
    return barrels

def create_frequency_based_barrels(lexicon: Dict, word_frequencies: Dict, 
                                   num_barrels: int = 10) -> List[Dict]:
    """
    Partition lexicon into barrels based on word frequency.
    
    Args:
        lexicon: Word -> word_id mapping
        word_frequencies: Word -> frequency mapping
        num_barrels: Number of barrels to create
        
    Returns:
        List of barrel dictionaries
    """
    if not lexicon:
        return []
    
    # Sort by frequency (descending)
    sorted_by_freq = sorted(
        [(word, lexicon[word], word_frequencies.get(word, 0)) 
         for word in lexicon],
        key=lambda x: x[2],  # Sort by frequency
        reverse=True
    )
    
    # Calculate total frequency and target per barrel
    total_freq = sum(freq for _, _, freq in sorted_by_freq)
    target_freq_per_barrel = total_freq / num_barrels
    
    barrels = []
    current_barrel = {}
    current_freq_sum = 0
    barrel_id = 1
    
    print(f"Partitioning by frequency: total frequency = {total_freq:,}")
    print(f"Target frequency per barrel: ~{target_freq_per_barrel:,.0f}")
    
    for word, word_id, freq in sorted_by_freq:
        current_barrel[word] = word_id
        current_freq_sum += freq
        
        # Create new barrel when frequency target reached or at end
        if (current_freq_sum >= target_freq_per_barrel and len(current_barrel) > 1) \
           or word == sorted_by_freq[-1][0]:
            
            barrel_words = list(current_barrel.keys())
            
            barrel_data = {
                "barrel_id": barrel_id,
                "range_start": barrel_words[0],
                "range_end": barrel_words[-1],
                "word_count": len(current_barrel),
                "frequency_sum": current_freq_sum,
                "lexicon": current_barrel.copy()
            }
            
            barrels.append(barrel_data)
            print(f"  Barrel {barrel_id}: {len(current_barrel):,} words, "
                  f"freq sum: {current_freq_sum:,.0f}")
            
            current_barrel = {}
            current_freq_sum = 0
            barrel_id += 1
    
    return barrels

def save_barrels(barrels: List[Dict], output_dir: str = "barrels"):
    """
    Save barrel files to disk.
    
    Args:
        barrels: List of barrel dictionaries
        output_dir: Directory to save barrel files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for barrel in barrels:
        filename = os.path.join(output_dir, f"barrel_{barrel['barrel_id']}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(barrel, f, indent=2)
    
    print(f"Saved {len(barrels)} barrel files to {output_dir}/")
    
    # Save barrel metadata
    metadata = {
        "total_barrels": len(barrels),
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "barrels": [
            {
                "barrel_id": b["barrel_id"],
                "range_start": b["range_start"],
                "range_end": b["range_end"],
                "word_count": b["word_count"]
            }
            for b in barrels
        ]
    }
    
    metadata_file = os.path.join(output_dir, "barrel_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved barrel metadata to {metadata_file}")

def load_barrels(barrels_dir: str) -> List[Dict]:
    """
    Load all barrel files from directory.
    
    Args:
        barrels_dir: Directory containing barrel files
        
    Returns:
        List of loaded barrel dictionaries
    """
    barrels = []
    
    if not os.path.exists(barrels_dir):
        print(f"Barrels directory not found: {barrels_dir}")
        return barrels
    
    # Find all barrel files
    barrel_files = [f for f in os.listdir(barrels_dir) 
                   if f.startswith("barrel_") and f.endswith(".json")]
    
    for filename in barrel_files:
        filepath = os.path.join(barrels_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                barrel = json.load(f)
            barrels.append(barrel)
        except Exception as e:
            print(f"Error loading barrel file {filename}: {e}")
    
    print(f"Loaded {len(barrels)} barrels from {barrels_dir}")
    return barrels

def merge_barrels(barrels: List[Dict]) -> Dict:
    """
    Merge multiple barrels back into a single lexicon.
    
    Args:
        barrels: List of barrel dictionaries
        
    Returns:
        Merged lexicon dictionary
    """
    merged_lexicon = {}
    
    for barrel in barrels:
        merged_lexicon.update(barrel.get("lexicon", {}))
    
    print(f"Merged {len(barrels)} barrels into lexicon with {len(merged_lexicon):,} words")
    return merged_lexicon

def get_barrel_for_word(word: str, barrels: List[Dict]) -> int:
    """
    Find which barrel contains a given word.
    
    Args:
        word: Word to search for
        barrels: List of barrel dictionaries
        
    Returns:
        Barrel ID containing the word, or -1 if not found
    """
    for barrel in barrels:
        if word in barrel.get("lexicon", {}):
            return barrel["barrel_id"]
    
    return -1

if __name__ == "__main__":
    print("Lexicon Partitioning Module")
    print("=" * 50)
    print("\nFunctions for partitioning lexicon into barrels for distributed processing.")
    
    # Example usage
    print("\nAVAILABLE FUNCTIONS:")
    print("1. create_alphabetical_barrels() - Partition alphabetically")
    print("2. create_size_based_barrels() - Partition by maximum size")
    print("3. create_frequency_based_barrels() - Partition by word frequency")
    print("4. save_barrels() - Save barrel files to disk")
    print("5. load_barrels() - Load barrels from disk")
    print("6. merge_barrels() - Merge barrels back into lexicon")
    print("7. get_barrel_for_word() - Find which barrel contains a word")