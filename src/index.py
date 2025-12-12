# index.py - Parallel indexing with memory-optimized POS tagging
import json as js 
import os
import csv
import time
import warnings
import gc
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore", message="generator ignored GeneratorExit")
warnings.filterwarnings("ignore", message=".*en_core_sci_sm.*")
warnings.filterwarnings("ignore", category=UserWarning)

from crawler import get_paper_batches, process_paper_batch, init_worker_nlp

# Base path
BASE_PATH = "D:/Cord19/cord/2022"

def generate_indexes_parallel(target_papers=50000, batch_size=500, num_workers=4, memory_safe=True):
    print("=" * 70)
    print(f"INCREMENTAL PARALLEL INDEXING (Target: {target_papers:,})")
    print("=" * 70)

    # --- Initialize Indexes Immediately (Not at the end) ---
    lexicon = {}
    forward_index = {}
    inverted_index = {}
    backward_index = {}
    word_id_counter = 1
    
    start_time = time.time()
    last_report_time = start_time
    total_processed = 0
    
    print(f"Starting stream with {num_workers} workers...")
    batch_gen = get_paper_batches(batch_size=batch_size, max_papers=target_papers)
    
    with Pool(processes=num_workers, initializer=init_worker_nlp) as pool:
        # Use imap_unordered for speed
        cursor = pool.imap_unordered(process_paper_batch, batch_gen)
        
        for batch_result in cursor:
            if not batch_result: continue
            
            # --- PROCESS DATA IMMEDIATELY ---
            # Instead of storing results, we update indexes and delete the data
            for paper in batch_result:
                if not paper or "tokens" not in paper: continue
                
                doc_id = paper["cord_uid"]
                tokens = paper["tokens"]
                
                # 1. Backward Index
                backward_index[doc_id] = tokens
                
                doc_word_ids = []
                word_freq = defaultdict(int)
                
                for token_info in tokens:
                    lemma = token_info['lemma']
                    pos = token_info.get('pos', 'UNK')
                    
                    # 2. Update Lexicon
                    if lemma not in lexicon:
                        lexicon[lemma] = {
                            "id": word_id_counter,
                            "pos_counts": defaultdict(int),
                            "lemma": lemma
                        }
                        word_id_counter += 1
                    
                    lexicon[lemma]["pos_counts"][pos] += 1
                    w_id = lexicon[lemma]["id"]
                    
                    doc_word_ids.append(w_id)
                    word_freq[w_id] += 1
                
                # 3. Update Forward Index
                forward_index[doc_id] = doc_word_ids
                
                # 4. Update Inverted Index
                for w_id, freq in word_freq.items():
                    if w_id not in inverted_index:
                        inverted_index[w_id] = {}
                    inverted_index[w_id][doc_id] = freq
                
                total_processed += 1

            # --- CRITICAL MEMORY FIX ---
            # Delete the processed batch from RAM immediately
            del batch_result
            
            # Progress Report
            current_time = time.time()
            if total_processed % 1000 == 0:
                elapsed = current_time - start_time
                rate = total_processed / elapsed
                remaining = (target_papers - total_processed) / rate if rate > 0 else 0
                time_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
                
                print(f"Indexed {total_processed:,} | Rate: {rate:.1f} docs/s | ETA: {time_str} | RAM Safe: Yes")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes")
    
    return lexicon, forward_index, inverted_index, backward_index


def save_index_files(lexicon, forward_index, inverted_index, backward_index, output_dir="indexes", total_time=None):
    """
    Save all index files to disk with statistics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        total_papers = len(forward_index)
        total_words = len(lexicon)
        
        print(f"\nSaving index files to '{output_dir}'...")
        print(f"Statistics: {total_papers:,} papers, {total_words:,} unique words")
        
        #save Lexicon with POS information
        lexicon_path = os.path.join(output_dir, "lexicon.json")
        
        #convert lexicon for JSON serialization
        lexicon_serializable = {}
        for word, info in lexicon.items():
            lexicon_serializable[word] = {
                "id": info["id"],
                "lemma": info.get("lemma", word),
                "pos_counts": dict(info["pos_counts"]) if isinstance(info["pos_counts"], defaultdict) else info["pos_counts"],
                "total_count": sum(info["pos_counts"].values())
            }
        
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            js.dump(lexicon_serializable, f, indent=2)
        print(f"✓ lexicon.json: {len(lexicon):,} words with POS distributions")

        #save Forward Index
        forward_path = os.path.join(output_dir, "forward_index.json")
        with open(forward_path, 'w', encoding='utf-8') as f:
            js.dump(forward_index, f, indent=None)
        print(f"forward_index.json: {len(forward_index):,} documents")
        
        #save Inverted Index
        inverted_path = os.path.join(output_dir, "inverted_index.json")
        with open(inverted_path, 'w', encoding='utf-8') as f:
            js.dump(inverted_index, f, indent=None)
        print(f"inverted_index.json: {len(inverted_index):,} terms")
        
        #save Backward Index (with POS information)
        backward_path = os.path.join(output_dir, "backward_index.json")
        with open(backward_path, 'w', encoding='utf-8') as f:
            js.dump(backward_index, f, indent=None)
        print(f"backward_index.json: {len(backward_index):,} documents")
        
        #save detailed stats
        stats = {
            "total_papers_indexed": total_papers,
            "total_unique_words": total_words,
            "inverted_index_terms": len(inverted_index),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_minutes": round(total_time // 60) if total_time else 0,
            "papers_per_second": round(total_papers / total_time, 1) if total_time else 0,
            "model_used": "en_core_sci_sm (scientific)",
            "index_sizes_bytes": {
                "lexicon": os.path.getsize(lexicon_path),
                "forward_index": os.path.getsize(forward_path),
                "inverted_index": os.path.getsize(inverted_path),
                "backward_index": os.path.getsize(backward_path)
            }
        }
        
        #pos summary for stats
        pos_summary = defaultdict(int)
        for word_info in lexicon.values():
            for pos, count in word_info["pos_counts"].items():
                pos_summary[pos] += count
        
        stats["pos_distribution"] = dict(sorted(pos_summary.items(), key=lambda x: x[1], reverse=True))
        
        stats_path = os.path.join(output_dir, "index_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            js.dump(stats, f, indent=2)
        print(f"index_statistics.json")
        
        print(f"\nall index files saved to {os.path.abspath(output_dir)}")
            
    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function with memory-optimized settings
    """
    print("=" * 70)
    print("CORD-19 INDEX GENERATION WITH SCIENTIFIC POS TAGGING")
    print("Optimized for memory usage with en_core_sci_sm")
    print("=" * 70)
    
    
    TARGET_PAPERS = 50000
    BATCH_SIZE = 50
    USE_PARALLEL = True
    MEMORY_SAFE = True  #enable memory optimizations
    NUM_WORKERS = 2    #fixed at 2 workers for scispaCy
    
    print(f"Target: Process {TARGET_PAPERS:,} papers")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mode: {'Parallel' if USE_PARALLEL else 'Single-threaded'}")
    print(f"Workers: {NUM_WORKERS} (optimized for scispaCy memory usage)")
    print(f"Model: en_core_sci_sm (scientific terminology)")
    print(f"Memory safe mode: {'Enabled' if MEMORY_SAFE else 'Disabled'}")
    print(f"Text preprocessing: URLs, emails, gibberish removal")
    print(f"POS tagging: enabled with scientific model")
    print("=" * 70)
    
    global start_time, total_time
    start_time = time.time()
    
    #generate indexes
    if USE_PARALLEL:
        lexicon, forward_index, inverted_index, backward_index = generate_indexes_parallel(
            target_papers=TARGET_PAPERS, 
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            memory_safe=MEMORY_SAFE
        )
    else:
        #single-threaded version would go here
        print("Single-threaded mode not implemented in this version")
        return
    
    #calculate total time
    total_time = time.time() - start_time
    
    #save indexes
    if lexicon:  #check if indexing was successful
        save_index_files(lexicon, forward_index, inverted_index, backward_index, 
                        output_dir="indexes", total_time=total_time)
    
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    
    #final statistics
    print("\n" + "=" * 70)
    print("Indexing Summary")
    print("=" * 70)
    print(f"Papers indexed: {len(forward_index):,}")
    print(f"Unique words: {len(lexicon):,}")
    print(f"Processing rate: {len(forward_index)/total_time:.1f} papers/sec")
    print(f"Model used: en_core_sci_sm (optimized for scientific text)")
    print(f"Index location: {os.path.abspath('indexes')}")
    print("=" * 70)

if __name__ == "__main__":
    main()