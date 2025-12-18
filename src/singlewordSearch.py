import json as js 
import pickle 
import struct 
import os 
import time 
# import pandas as pd  <-- REMOVED PANDAS TO SAVE RAM
import re
import concurrent.futures
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import csv
import sys

# --- FIX FOR OVERFLOW ERROR ON WINDOWS ---
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

BARREL_MAP_PATH = os.path.join(DATA_DIR, "barrels", "barrel_mappings.json")
BARRELS_DIR = os.path.join(DATA_DIR, "barrels")
COMPRESSED_BARRELS_DIR = os.path.join(DATA_DIR, "compressed_barrels")
DOC_MAP_PATH = os.path.join(COMPRESSED_BARRELS_DIR, "doc_id_mapping.pkl")
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
BACKWARD_INDEX_PATH = os.path.join(INDEXES_DIR, "backward_index.json")
LEXICON_PATH = os.path.join(INDEXES_DIR, "lexicon.json")
TRIE_PATH = os.path.join(DATA_DIR, "barrel_trie.pkl")

dynamic_indexer_ref = None

class TrieNode:
    __slots__ = ('children', 'barrel_ids', 'is_end_of_word')
    def __init__(self):
        self.children = {}  
        self.barrel_ids = set()
        self.is_end_of_word = False

class BarrelTrie:
    def __init__(self):
        self.root = TrieNode()
        self.word_to_barrel_cache = {}
        self.size = 0
    
    def insert(self, word: str, barrel_id: int):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.barrel_ids.add(barrel_id)
        node.is_end_of_word = True
        node.barrel_ids.add(barrel_id)
        if word not in self.word_to_barrel_cache:
            self.word_to_barrel_cache[word] = set()
        self.word_to_barrel_cache[word].add(barrel_id)
        self.size += 1
    
    def get_barrels_for_word(self, word: str) -> List[int]:
        if word in self.word_to_barrel_cache:
            return list(self.word_to_barrel_cache[word])
        node = self.root
        for char in word:
            if char not in node.children: return []
            node = node.children[char]
        if node.is_end_of_word:
            result = list(node.barrel_ids)
            self.word_to_barrel_cache[word] = set(result)
            return result
        return []
    
    def save_to_file(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load_from_file(filepath: str) -> Optional['BarrelTrie']:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except: pass
        return None

class OptimizedBarrelLookup:
    def __init__(self):
        self.trie = None
        self._barrel_cache = {}
        self._last_access_time = {}
        self._max_cache_size = 20
        
    def load_trie(self):
        self.trie = BarrelTrie.load_from_file(TRIE_PATH)
        print("Building trie from barrel mappings...")
        start_time = time.time()
        try:
            with open(BARREL_MAP_PATH, 'r') as f:
                mappings = js.load(f)
            self.trie = BarrelTrie()
            if "word_to_barrel" in mappings:
                for word, barrel_id in mappings["word_to_barrel"].items():
                    self.trie.insert(word.lower(), barrel_id)
            self.trie.save_to_file(TRIE_PATH)
            print(f"Trie with {self.trie.size} words was built in {time.time() - start_time:.2f}s")
            return True
        except Exception as e:
            print(f"Trie couldn't be built: {e}")
            self.trie = None
            return False
    
    def get_barrels_for_word(self, word: str) -> List[str]:
        if not self.trie:
            if not self.load_trie(): return []
        barrel_ids = self.trie.get_barrels_for_word(word.lower())
        return [str(bid) for bid in barrel_ids] if barrel_ids else []
    
    def get_barrel_data(self, barrel_id: int) -> Dict:
        if barrel_id in self._barrel_cache:
            self._last_access_time[barrel_id] = time.time()
            return self._barrel_cache[barrel_id]
        try:
            barrel_path = os.path.join(COMPRESSED_BARRELS_DIR, f"compressed_barrel_{barrel_id}.pkl")
            if not os.path.exists(barrel_path):
                barrel_path = os.path.join(COMPRESSED_BARRELS_DIR, f"compressed_barrel_{barrel_id}.pickle")
                if not os.path.exists(barrel_path): return {}
            with open(barrel_path, 'rb') as f:
                barrel_data = pickle.load(f)
            if len(self._barrel_cache) >= self._max_cache_size:
                oldest = min(self._last_access_time.items(), key=lambda x: x[1])[0]
                del self._barrel_cache[oldest]
                del self._last_access_time[oldest]
            self._barrel_cache[barrel_id] = barrel_data
            self._last_access_time[barrel_id] = time.time()
            return barrel_data
        except Exception: return {}

barrel_lookup = OptimizedBarrelLookup()
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_corpus.csv")

# ============================================================================
# OPTIMIZED DOCUMENT MANAGER (Replaced Pandas with CSV + Filtering)
# ============================================================================
class DocumentManager:
    def __init__(self):
        self.title_cache = {} # Dict[str, str]
        self.dynamic_docs = {}
        self.loaded = False
        # NEW: Track total raw papers processed (for display purposes)
        self.total_docs_in_corpus = 0 
    
    def add_dynamic_doc(self, doc_id, title, content):
        self.dynamic_docs[doc_id] = {'title': title, 'content': content}
        self.title_cache[doc_id] = title

    def load_metadata(self):
        """
        Loads titles ONLY for documents that exist in the search index.
        Saves ~500MB of RAM by ignoring unused titles.
        Also counts TOTAL lines to show impressive stats on frontend.
        """
        if self.loaded: return True
        if os.path.exists(PROCESSED_DATA_PATH):
            print(f"Loading corpus headers (Filtered) from {PROCESSED_DATA_PATH}...")
            
            # 1. Load valid IDs first
            valid_ids = set()
            if os.path.exists(DOC_MAP_PATH):
                try:
                    with open(DOC_MAP_PATH, 'rb') as f:
                        mapping = pickle.load(f)
                        valid_ids = set(mapping.get("str_to_int", {}).keys())
                    print(f"Filter active: Only loading titles for {len(valid_ids)} indexed docs.")
                except:
                    pass

            try:
                # 2. Stream CSV and filter
                with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.DictReader(f)
                    total_count = 0
                    
                    for row in reader:
                        total_count += 1 # Count every single row
                        
                        doc_id = row.get('id') or row.get('cord_uid')
                        if doc_id:
                            doc_id = str(doc_id).strip()
                            # Only store if it's in our index (or if we have no index map)
                            if not valid_ids or doc_id in valid_ids:
                                self.title_cache[doc_id] = row.get('title', 'Untitled')
                    
                    self.total_docs_in_corpus = total_count
                            
                self.loaded = True
                print(f"RAM Optimized: Loaded {len(self.title_cache)} relevant titles.")
                print(f"Total processed documents found: {self.total_docs_in_corpus}")
                return True
            except Exception as e:
                print(f"Error loading corpus: {e}")
                return False
        return False
    
    def get_document_title(self, doc_id: str) -> str:
        if doc_id in self.dynamic_docs:
            return self.dynamic_docs[doc_id]['title']
        if not self.loaded: self.load_metadata()
        return self.title_cache.get(str(doc_id), "Untitled Document")
    
    def get_document_text(self, doc_id: str) -> str:
        # 1. Check dynamic docs
        if doc_id in self.dynamic_docs:
            return self.dynamic_docs[doc_id]['content']
        # 2. Check disk (Streaming read)
        try:
            with open(PROCESSED_DATA_PATH, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                target = str(doc_id).strip()
                for row in reader:
                    curr = str(row.get('id') or row.get('cord_uid')).strip()
                    if curr == target:
                        return row.get('content', "No content available.")
            return "Document not found."
        except Exception as e:
            return f"Error reading document: {e}"

doc_manager = DocumentManager()

def verify_paths():
    """Fast path verification"""
    required = [BARREL_MAP_PATH, COMPRESSED_BARRELS_DIR, DOC_MAP_PATH, PROCESSED_DATA_PATH]
    print("\n" + "=" * 60 + "\nPATH VERIFICATION\n" + "=" * 60)
    all_exist = True
    for path in required:
        exists = os.path.exists(path)
        print(f"{'Exists!' if exists else 'Missing!'} {os.path.basename(path)}")
        if not exists: all_exist = False
    print("=" * 60)
    return all_exist
# ============================================================================
# decompressing functions (optimised but mostly similar to previous)
# ============================================================================
def varbyte_decode(byte_arr: bytes) -> List[int]:
    """Optimized varbyte decoding"""
    numbers = []
    curr = 0

    for byte in byte_arr:
        if byte < 128:
            curr = (curr << 7) | byte  #Faster than 128 * curr + byte (was used in barrels)
        else:
            curr = (curr << 7) | (byte - 128)
            numbers.append(curr)
            curr = 0
    
    return numbers

def decompress_posting_list(compressed_data: Dict) -> Tuple[List[int], Optional[List[int]]]:
    data_type = compressed_data.get("type", "empty")
    
    if data_type == "empty":
        return [], None
    
    elif data_type == "no_freqs":
        encoded_gaps = compressed_data["data"]
        gaps = varbyte_decode(encoded_gaps)
        
        if not gaps:
            return [], None
        
        # Build doc_ids from gaps
        doc_ids = [0] * len(gaps)
        doc_ids[0] = gaps[0]
        for i in range(1, len(gaps)):
            doc_ids[i] = doc_ids[i-1] + gaps[i]
        
        return doc_ids, None
    
    elif data_type == "with_freqs":
        data = compressed_data["data"]
        
        # Unpack using memory views for speed
        gap_len = struct.unpack_from('I', data, 0)[0]
        avg_freq = struct.unpack_from('d', data, 4 + gap_len)[0]
        freq_len = struct.unpack_from('I', data, 4 + gap_len + 8)[0]
        
        #Decode gaps and frequencies
        gaps = varbyte_decode(data[4:4 + gap_len])
        freqs_diffs = varbyte_decode(data[4 + gap_len + 8 + 4:4 + gap_len + 8 + 4 + freq_len])
        
        if not gaps:
            return [], []
        
        #Build results
        doc_ids = [0] * len(gaps)
        doc_ids[0] = gaps[0]
        for i in range(1, len(gaps)):
            doc_ids[i] = doc_ids[i-1] + gaps[i]
        
        frequencies = [int(avg_freq + diff) for diff in freqs_diffs]
        return doc_ids, frequencies
    
    return [], None

# ============================================================================
# search function sections (optimized with trie and parallelism)
# ============================================================================
def search_in_barrel(barrel_data: Dict, word: str) -> Tuple[List[int], Optional[List[int]]]:
    """Fast in-barrel search"""
    try:
        compressed_words = barrel_data.get("compressed_words", {})
        if word in compressed_words:
            compressed = compressed_words[word]["compressed_data"]
            return decompress_posting_list(compressed)
    except Exception:
        pass
    
    return [], None

def search_word_parallel(word: str) -> Tuple[List[int], Optional[List[int]]]:
    barrel_ids = barrel_lookup.get_barrels_for_word(word)
    
    if not barrel_ids:
        return [], None

    all_doc_ids = []
    all_frequencies = []
    
    # Use ThreadPool for parallel barrel loading and searching 
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(barrel_ids))) as executor:
        future_to_barrel = {}
        for barrel_id_str in barrel_ids:
            try:
                barrel_id = int(barrel_id_str)
                future = executor.submit(
                    lambda bid: (bid, barrel_lookup.get_barrel_data(bid)),
                    barrel_id
                )
                future_to_barrel[future] = barrel_id
            except ValueError:
                continue
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_barrel):
            barrel_id = future_to_barrel[future]
            try:
                barrel_data = future.result()[1]  # Get the barrel data
                if barrel_data:
                    doc_ids, frequencies = search_in_barrel(barrel_data, word)
                    if doc_ids:
                        all_doc_ids.extend(doc_ids)
                        if frequencies:
                            all_frequencies.extend(frequencies)
            except Exception:
                continue
    
    return all_doc_ids, all_frequencies if all_frequencies else None

def search_word(word: str, use_compressed: bool = True) -> Tuple[List[str], Optional[List[int]]]:
    """Main search function - Hybrid (Barrels + Dynamic Index)"""
    start_time = time.time()
    
    # 1. Search Static Barrels
    # Note: This returns INTEGERS (mapped IDs)
    barrel_doc_ids, barrel_freqs = search_word_parallel(word)
    
    # 2. Search Dynamic Index (The "Gap Bridge")
    global dynamic_indexer_ref
    dyn_doc_ids = []
    dyn_freqs = []
    
    if dynamic_indexer_ref:
        # This returns STRINGS (actual IDs)
        dyn_doc_ids, dyn_freqs = dynamic_indexer_ref.search_dynamic_word(word)
    
    elapsed_time = time.time() - start_time
    
    # 3. Merge Results
    # We must ensure all IDs are consistent. 
    # Since barrel IDs are ints, we leave them as is for now, resolve_document_ids handles them.
    # Dynamic IDs are strings, we add them to the list.
    
    final_ids = []
    final_freqs = []
    
    if barrel_doc_ids:
        final_ids.extend(barrel_doc_ids)
        if barrel_freqs: final_freqs.extend(barrel_freqs)
    
    if dyn_doc_ids:
        final_ids.extend(dyn_doc_ids)
        if dyn_freqs: final_freqs.extend(dyn_freqs)

    if final_ids:
        print(f"-- Found {len(final_ids)} documents in {elapsed_time:.3f}s")
    
    return final_ids, final_freqs

# ============================================================================
# doc resolution functions (optimized with caching)
# ============================================================================

_doc_mapping_cache = None

def resolve_document_ids(int_doc_ids: List[int]) -> List[str]:
    """Fast document ID resolution with caching"""
    global _doc_mapping_cache
    
    if not int_doc_ids:
        return []
    
    # Load mapping once and cache it
    if _doc_mapping_cache is None:
        try:
            if os.path.exists(DOC_MAP_PATH):
                with open(DOC_MAP_PATH, 'rb') as f:
                    mapping_data = pickle.load(f)
                _doc_mapping_cache = mapping_data.get("int_to_str", {})
            else:
                _doc_mapping_cache = {}
        except Exception:
            _doc_mapping_cache = {}
    
    # Fast batch resolution
    if _doc_mapping_cache:
        return [_doc_mapping_cache.get(doc_id, str(doc_id)) for doc_id in int_doc_ids]
    else:
        return [str(doc_id) for doc_id in int_doc_ids]

# ============================================================================
# lemma funcs section (optimized with caching)
# ============================================================================
_lexicon_cache = None

def get_lemma_for_word(word: str, return_all_variations: bool = False):
    """Fast lemma lookup with caching"""
    global _lexicon_cache
    
    if _lexicon_cache is None:
        try:
            if os.path.exists(LEXICON_PATH):
                with open(LEXICON_PATH, 'r') as f:
                    _lexicon_cache = js.load(f)
            else:
                _lexicon_cache = {}
        except Exception:
            _lexicon_cache = {}
    
    word_lower = word.lower()
    
    # Fast lookup
    if word in _lexicon_cache:
        entry = _lexicon_cache[word]
        if isinstance(entry, dict) and "lemma" in entry:
            if return_all_variations:
                return [entry["lemma"]]
            return entry["lemma"]
    
    # isn't affected by case
    for key, value in _lexicon_cache.items():
        if key.lower() == word_lower:
            if isinstance(value, dict) and "lemma" in value:
                if return_all_variations:
                    return [value["lemma"]]
                return value["lemma"]
    
    # fallback (basically return the word itself)
    if return_all_variations:
        return [word]
    return word

# ============================================================================
# display functions for all search results (optimized for speed)
# ============================================================================
def display_search_results(word: str, doc_ids: List[int], frequencies: Optional[List[int]] = None, max_results: int = 10):
    """minimal overhead display of search results"""
    if not doc_ids:
        print(f"\nNo documents found containing '{word}'")
        return
    
    total_docs = len(doc_ids)
    display_count = min(max_results, total_docs)
    
    # Convert IDs in batch
    str_doc_ids = resolve_document_ids(doc_ids[:display_count])
    
    print(f"\n{'='*80}")
    print(f"TOP {display_count} RESULTS FOR: '{word.upper()}' (of {total_docs})")
    print(f"{'='*80}\n")
    
    # Batch title fetching
    titles = {}
    for doc_id in str_doc_ids:
        titles[doc_id] = doc_manager.get_document_title(doc_id)
    
    # loop to display
    for i, str_id in enumerate(str_doc_ids, 1):
        title = titles[str_id]
        
        # shorten long titles using ellipsis
        if len(title) > 70:
            title = title[:67] + "..."
        
        # Frequency info
        freq_info = ""
        if frequencies and i <= len(frequencies):
            freq_info = f" | Freq: {frequencies[i-1]}"
        
        # Display
        print(f"{i:2d}. {title}")
        print(f"    ID: {str_id}{freq_info}")
        
        # Only show preview for top 3 results to save time (can be changed)
        if i <= 3:
            # Simple preview from title
            if "Untitled" not in title:
                print(f"{title[:100]}...")
        
        print()
    
    # Quick summary
    print(f"{'─'*40}")
    if frequencies and len(frequencies) >= display_count:
        total_freq = sum(frequencies[:display_count])
        print(f"Total frequency in top {display_count}: {total_freq}")
    print(f"Showing {display_count} of {total_docs} documents")
    print(f"{'='*80}")

# ============================================================================
# test performance (use this section when submitting assignment proof)
# ============================================================================
def performance_test():
    """Test search performance"""
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE TESTING")
    print("=" * 80)
    
    # Load trie first
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    test_words = ["covid", "vaccine", "coronavirus", "treatment", "pandemic"]
    
    print("\nTesting single word searches:\n")
    
    for word in test_words:
        start_time = time.time()
        doc_ids, frequencies = search_word(word)
        elapsed = time.time() - start_time
        
        if doc_ids:
            status = "performance was optimal" if elapsed < 0.5 else "performance was suboptimal"
            print(f"{status} '{word}': {len(doc_ids)} docs in {elapsed:.3f}s")
            
            if elapsed > 0.5:
                print(f"Above 500ms target: {elapsed:.3f}s")
        else:
            print(f"'{word}': No results in {elapsed:.3f}s")
    
    print(f"\n{'=' * 80}")
    print("performance test complete")
    print("=" * 80)

def test_basic_search():
    print("\n" + "=" * 80)
    print("searching basic test")
    print("=" * 80)
    
    # Load necessary components
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    test_words = ["covid", "vaccine", "coronavirus"]
    
    for word in test_words:
        print(f"\n{'─' * 60}")
        print(f"Testing: '{word}'")
        
        start_time = time.time()
        doc_ids, frequencies = search_word(word)
        elapsed = time.time() - start_time
        
        if doc_ids:
            print(f"Found {len(doc_ids)} docs in {elapsed:.3f}s")
            display_search_results(word, doc_ids[:5], frequencies, max_results=3)
        else:
            print(f"No results in {elapsed:.3f}s")
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print("=" * 80)

# ============================================================================
# INTERACTIVE SEARCH
# ============================================================================
def interactive_search():
    """Fast interactive search"""
    print("\n" + "=" * 80)
    print("Interactive User Search")
    print("=" * 80)
    print("Target: < 500ms per query")
    print("=" * 80)
    
    # Initialize
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    while True:
        try:
            user_input = input("\nSearch word (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("You chose to exit.")
                break
            
            if not user_input:
                continue
            
            #performance metric measurement
            start_time = time.time()
            doc_ids, frequencies = search_word(user_input)
            elapsed = time.time() - start_time
            
            if doc_ids:
                print(f"\nFound {len(doc_ids)} docs in {elapsed:.3f}s")
                
                if elapsed > 0.5:
                    print(f"Search took {elapsed:.3f}s (target: < 0.5s)")
                
                display_search_results(user_input, doc_ids, frequencies, max_results=10)
            else:
                print(f"\nNo documents found in {elapsed:.3f}s")
                
        except KeyboardInterrupt:
            print("\nSearch cancelled")
            break
        except Exception as e:
            print(f"\nError: {str(e)[:100]}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Search Module - Single Word Search")
    print("=" * 80)
    
    # Verify paths
    if not verify_paths():
        print("\nSome required files could not be accessed.")
        response = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            exit(1)
    
    # Menu
    print("\nSelect mode:")
    print("1. Performance test")
    print("2. Basic search test")
    print("3. Interactive search")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        performance_test()
    elif choice == "2":
        test_basic_search()
    elif choice == "3":
        interactive_search()
    else:
        print("Search complete!")