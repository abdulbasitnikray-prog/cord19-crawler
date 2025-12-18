import json
import time
import gc

class TrieNode:
    __slots__ = ('children', 'is_end', 'word', 'frequency', 'top_cache')
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None
        self.frequency = 0
        self.top_cache = [] 

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
            
            # OPTIMIZATION: Take top 25,000 words. 
            # This is the sweet spot for RAM vs Utility.
            TOP_N = 25000
            
            sorted_words = sorted(
                [(k, v.get("total_count", 0)) for k, v in lexicon.items() if isinstance(v, dict)], 
                key=lambda x: x[1], 
                reverse=True
            )[:TOP_N]

            for word, freq in sorted_words:
                if len(word) > 2: 
                    self.insert(word, freq)
                
            self.loaded = True
            print(f"Autocomplete Ready: {len(sorted_words)} words in {time.time()-start:.2f}s")
            
            # Force cleanup
            del lexicon
            del sorted_words
            gc.collect()
            
        except Exception as e:
            print(f"Failed to load lexicon: {e}")

    def insert(self, word, freq):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
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