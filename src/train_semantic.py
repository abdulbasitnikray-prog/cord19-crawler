import os
import time
import logging
from gensim.models import Word2Vec
from crawler import get_paper_batches, process_paper_batch, init_worker_nlp

# --- CONFIGURATION ---
MODEL_OUTPUT_FILE = "cord19_semantic.model"
TARGET_PAPERS = 50000 
WORKERS = 4
# ---------------------

# Setup logging to see progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Cord19Sentences:
    """
    Memory-efficient iterator that streams papers, processes them, 
    and yields token lists to Gensim for training.
    """
    def __init__(self):
        self.count = 0

    def __iter__(self):
        # 1. Get batches of raw papers using your existing crawler logic
        batch_generator = get_paper_batches(batch_size=1000, max_papers=TARGET_PAPERS)

        for i, batch in enumerate(batch_generator):
            print(f"Training iterator : Loading Batch {i}")
            # 2. Process batch to get Lemmas (using optimized spaCy pipe)
            processed_papers = process_paper_batch(batch)
            
            # 3. Yield sentences
            for paper in processed_papers:
                if 'tokens' in paper:
                    # Extract just the lemmas: ['covid', 'virus', 'infection']
                    tokens = [t['lemma'] for t in paper['tokens']]
                    
                    # Filter out short/empty lines
                    if len(tokens) > 5:
                        self.count += 1
                        yield tokens

def train_model():
    print("="*60)
    print("STARTING SEMANTIC MODEL TRAINING")
    print("="*60)
    start_time = time.time()

    # 1. Initialize the data stream
    sentences = Cord19Sentences()

    # 2. Initialize and Train Word2Vec
    # vector_size=100:  Dimensions. 100 is standard for this dataset size.
    # window=5:         Context window (5 words left/right).
    # min_count=10:     Ignore rare typos to keep model small.
    # epochs=1:         Number of times to go through the data.
    print(f"Training Word2Vec model on {TARGET_PAPERS} papers...")
    
    model = Word2Vec(
        sentences=sentences, 
        vector_size=100, 
        window=5, 
        min_count=10, 
        workers=WORKERS,
        epochs=1
    )

    # 3. Save the model
    print(f"\nSaving model to {MODEL_OUTPUT_FILE}...")
    model.save(MODEL_OUTPUT_FILE)

    total_time = time.time() - start_time
    print("="*60)
    print(f"TRAINING COMPLETE in {total_time/60:.2f} minutes")
    print(f"Model saved: {os.path.abspath(MODEL_OUTPUT_FILE)}")
    
if __name__ == "__main__":
    # Initialize shared memory for NLP if needed
    init_worker_nlp()
    train_model()