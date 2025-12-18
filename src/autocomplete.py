import json
import time

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None
        self.frequency = 0
        self.top_cache = [] # Optimization: Cache top results at every node

class AutocompleteEngine:
    def __init__(self):
        self.root = TrieNode()
        self.loaded = False

    def load_from_lexicon(self, lexicon_path):
        print(f"Loading Autocomplete Trie from {lexicon_path}...")
        start = time.time()
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
            
            # --- OPTIMIZATION: Sort by frequency and take only top 50k ---
            print(f"Sorting {len(lexicon)} words by frequency...")
            
            # Convert dict to list of (word, freq)
            word_list = []
            for word, data in lexicon.items():
                freq = data.get("total_count", 0) if isinstance(data, dict) else 0
                word_list.append((word, freq))
            
            # Sort descending
            word_list.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top 50,000
            TOP_N = 50000
            pruned_list = word_list[:TOP_N]
            
            print(f"Pruned to top {TOP_N} words. Building Trie...")

            for word, freq in pruned_list:
                # Skip tiny words to save more RAM
                if len(word) > 2:
                    self.insert(word, freq)
                
            self.loaded = True
            print(f"Autocomplete Ready: {len(pruned_list)} words in {time.time()-start:.2f}s")
            
            # Force garbage collection
            del lexicon
            del word_list
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Failed to load lexicon for autocomplete: {e}")

    def insert(self, word, freq):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Cache top 5 high-frequency words at this node
            node.top_cache.append((freq, word))
            node.top_cache.sort(key=lambda x: x[0], reverse=True)
            if len(node.top_cache) > 5:
                node.top_cache = node.top_cache[:5]
            
        node.is_end = True
        node.word = word
        node.frequency = freq

    def search(self, prefix):
        if not prefix or not self.loaded: return []
        node = self.root
        for char in prefix.lower():
            if char not in node.children: return []
            node = node.children[char]
        return [item[1] for item in node.top_cache]