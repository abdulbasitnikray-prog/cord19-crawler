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
        """Loads the pre-trained Word2Vec model"""
        if os.path.exists(MODEL_PATH):
            print(f"Loading Semantic Model from {MODEL_PATH}...")
            start = time.time()
            try:
                self.model = Word2Vec.load(MODEL_PATH)
                self.is_loaded = True
                print(f"Semantic Model loaded in {time.time() - start:.2f}s")
            except Exception as e:
                print(f"Error loading semantic model: {e}")
        else:
            print(f"Semantic model file not found at {MODEL_PATH}")

    def get_similar_words(self, word, top_n=3):
        """Returns a list of semantically similar words"""
        if not self.is_loaded or not self.model:
            return []
        
        # Clean word
        word = word.lower().strip()
        
        try:
            # Check if word is in vocabulary
            if word in self.model.wv:
                # Get most similar words (returns tuples of (word, similarity))
                similar = self.model.wv.most_similar(word, topn=top_n)
                # Return just the words
                return [s[0] for s in similar]
            else:
                return []
        except KeyError:
            return []
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def expand_query(self, query_words):
        """
        Takes a list of query words and adds synonyms.
        Returns a dict mapping original words to their expansions.
        """
        expansion_map = {}
        for word in query_words:
            synonyms = self.get_similar_words(word, top_n=2)
            if synonyms:
                expansion_map[word] = synonyms
        return expansion_map

# Global instance
semantic_engine = SemanticSearchEngine()