import os
import time
from gensim.models import Word2Vec

# Configuration
MODEL_PATH = "cord19_semantic.model"

class SemanticSearchEngine:
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Loads the pre-trained Word2Vec model with memory mapping"""
        if os.path.exists(MODEL_PATH):
            print(f"Loading Semantic Model (Mmap) from {MODEL_PATH}...")
            start = time.time()
            try:
                # OPTIMIZATION: mmap='r' keeps model on disk, saving ~300MB RAM
                self.model = Word2Vec.load(MODEL_PATH, mmap='r')
                self.is_loaded = True
                print(f"Semantic Model loaded in {time.time() - start:.2f}s")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
        else:
            print(f"Semantic model file not found at {MODEL_PATH}")

    def get_similar_words(self, word, top_n=3):
        if not self.is_loaded or not self.model: return []
        word = word.lower().strip()
        try:
            if word in self.model.wv:
                similar = self.model.wv.most_similar(word, topn=top_n)
                return [s[0] for s in similar]
            return []
        except: return []

    def expand_query(self, query_words):
        expansion_map = {}
        for word in query_words:
            synonyms = self.get_similar_words(word, top_n=2)
            if synonyms:
                expansion_map[word] = synonyms
        return expansion_map

semantic_engine = SemanticSearchEngine()