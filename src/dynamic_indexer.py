import os
import json
import time
from collections import defaultdict
from crawler import process_paper_single

# Path for the isolated dynamic index
DYNAMIC_INDEX_PATH = os.path.join("data", "indexes", "dynamic_index.json")

class DynamicIndexer:
    """
    Manages a lightweight 'Delta Index' for new documents.
    Does NOT load the massive static index, ensuring instant updates.
    """
    
    def __init__(self, index_dir="data/indexes"):
        self.index_dir = index_dir
        self.lexicon = {}          # {word: {id: 1, doc_counts: {...}}}
        self.forward_index = {}    # {doc_id: [word_id, ...]}
        self.inverted_index = {}   # {word_id: {doc_id: freq}}
        self.word_id_counter = 1
        self.loaded = False
        
        # Ensure directory exists
        os.makedirs(index_dir, exist_ok=True)
        self.load_dynamic_index()

    def load_dynamic_index(self):
        """Loads ONLY the dynamic additions, not the huge static corpus"""
        if os.path.exists(DYNAMIC_INDEX_PATH):
            try:
                print(f"Loading dynamic index from {DYNAMIC_INDEX_PATH}...")
                with open(DYNAMIC_INDEX_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.lexicon = data.get("lexicon", {})
                    self.forward_index = data.get("forward_index", {})
                    # Convert keys back to integers for inverted index
                    self.inverted_index = {int(k): v for k, v in data.get("inverted_index", {}).items()}
                    self.word_id_counter = data.get("word_id_counter", 1)
                print(f"✓ Dynamic Index loaded: {len(self.forward_index)} new docs")
            except Exception as e:
                print(f"⚠ Corrupt dynamic index, starting fresh: {e}")
                self.lexicon = {}
        self.loaded = True

    def save_dynamic_index(self):
        """Saves only the delta data. Instant operation."""
        data = {
            "lexicon": self.lexicon,
            "forward_index": self.forward_index,
            "inverted_index": self.inverted_index,
            "word_id_counter": self.word_id_counter
        }
        with open(DYNAMIC_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"✓ Dynamic index saved ({len(self.forward_index)} docs)")

    def add_single_document(self, document_data):
        """
        Adds a document to the delta index.
        Time complexity: O(Words in Doc) -> ~0.05 seconds
        """
        doc_id = document_data["cord_uid"]
        
        if doc_id in self.forward_index:
            return False # Already exists in dynamic index

        print(f"Indexing new doc: {doc_id}")
        
        # 1. Process Text (NLP)
        json_parse = {
            "metadata": {"title": document_data.get("title", "")},
            "body_text": [{"text": document_data["content"]}]
        }
        processed = process_paper_single(json_parse, cord_uid=doc_id)
        
        if not processed or "tokens" not in processed:
            return False
        
        tokens = processed["tokens"]
        doc_word_ids = []
        word_freq = {}
        
        # 2. Update In-Memory Indexes
        for token_info in tokens:
            word = token_info['lemma']
            
            # Assign Dynamic ID
            if word not in self.lexicon:
                self.lexicon[word] = {"id": self.word_id_counter}
                self.word_id_counter += 1
            
            w_id = self.lexicon[word]["id"]
            doc_word_ids.append(w_id)
            word_freq[w_id] = word_freq.get(w_id, 0) + 1
            
        self.forward_index[doc_id] = doc_word_ids
        
        for w_id, freq in word_freq.items():
            if w_id not in self.inverted_index:
                self.inverted_index[w_id] = {}
            self.inverted_index[w_id][doc_id] = freq
            
        # 3. Auto-Save (Since it's small, we save on every add for safety)
        self.save_dynamic_index()
        return True

    def search_dynamic_word(self, word):
        """Searches only the dynamic layer"""
        word = word.lower().strip()
        if word in self.lexicon:
            w_id = self.lexicon[word]["id"]
            if w_id in self.inverted_index:
                # Return keys (doc_ids) and values (frequencies)
                return list(self.inverted_index[w_id].keys()), list(self.inverted_index[w_id].values())
        return [], []