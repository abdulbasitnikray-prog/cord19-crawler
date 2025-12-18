import json
import time
import heapq

class TrieNode:
    __slots__ = ('children', 'is_end', 'word', 'frequency', 'top_cache')  # Memory optimization
    
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None
        self.frequency = 0
        self.top_cache = []  # Optimization: Cache top results at every node

class AutocompleteEngine:
    def __init__(self):
        self.root = TrieNode()
        self.loaded = False

    def load_from_lexicon(self, lexicon_path):
        print("Loading Autocomplete Trie...")
        start = time.time()
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
            
            for word, data in lexicon.items():
                freq = data.get("total_count", 0) if isinstance(data, dict) else 0
                self.insert(word, freq)
                
            self.loaded = True
            print(f"Autocomplete Ready: {len(lexicon)} words in {time.time()-start:.2f}s")
        except Exception as e:
            print(f"Failed to load lexicon for autocomplete: {e}")

    def insert(self, word, freq):
        node = self.root
        word_lower = word.lower()  # Cache lowercase conversion
        for char in word_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Use heap for efficient top-k maintenance
            import heapq
            if len(node.top_cache) < 5:
                heapq.heappush(node.top_cache, (freq, word))
            elif freq > node.top_cache[0][0]:
                heapq.heapreplace(node.top_cache, (freq, word))
            
        node.is_end = True
        node.word = word
        node.frequency = freq

    def search(self, prefix):
        if not prefix or not self.loaded:
            return []
        node = self.root
        prefix_lower = prefix.lower()  # Cache lowercase
        for char in prefix_lower:
            if char not in node.children:
                return []
            node = node.children[char]
        # Return sorted results from heap (largest first)
        import heapq
        return [item[1] for item in heapq.nlargest(5, node.top_cache)]