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

def build_indexes_from_batch_results(batch_results, target_papers=50000):
    """
    Build all indexes from processed batch results with integrated POS information
    """
    print(f"\n--- Building Indexes from Processed Results (Target: {target_papers:,} papers) ---")
    
    # full lexicon with POS information
    lexicon = {}           # word -> {"id": word_id, "pos_counts": {pos: count}, "lemma": lemma}
    forward_index = {}     # doc_id -> [word_id1, word_id2, ...]
    inverted_index = {}    # word_id -> {doc_id: frequency}
    backward_index = {}    # doc_id -> [{word, lemma, pos, tag}, ...]
    word_id_counter = 1
    
    total_papers = 0
    start_time = time.time()
    last_report_time = start_time
    
    # process all batch results
    for batch_idx, batch in enumerate(batch_results):
        for paper_data in batch:
            if not paper_data or "tokens" not in paper_data:
                continue
                
            doc_id = paper_data["cord_uid"]
            tokens = paper_data["tokens"]
            
            # update backward index with POS information
            backward_index[doc_id] = tokens
            
            # update lexicon and forward index
            doc_word_ids = []
            word_freq = defaultdict(int)
            
            for token_info in tokens:
                lemma = token_info['lemma']
                pos = token_info['pos']
                tag = token_info['tag']
                
                # use lemma as the word key
                word_key = lemma
                
                # add to lexicon if new
                if word_key not in lexicon:
                    lexicon[word_key] = {
                        "id": word_id_counter,
                        "pos_counts": defaultdict(int),
                        "lemma": lemma
                    }
                    word_id_counter += 1
                
                # update POS counts for this word
                lexicon[word_key]["pos_counts"][pos] += 1
                
                word_id = lexicon[word_key]["id"]
                doc_word_ids.append(word_id)
                word_freq[word_id] += 1
            
            # update forward index
            forward_index[doc_id] = doc_word_ids
            
            # update inverted index
            for word_id, freq in word_freq.items():
                if word_id not in inverted_index:
                    inverted_index[word_id] = {}
                inverted_index[word_id][doc_id] = freq
            
            total_papers += 1
            
            # progress reporting every 1000 papers or 30 seconds
            current_time = time.time()
            if total_papers % 1000 == 0 or (current_time - last_report_time) > 30:
                elapsed = current_time - start_time
                rate = total_papers / elapsed
                
                # to find out how much time remains 
                papers_remaining = target_papers - total_papers
                if rate > 0:
                    time_remaining = papers_remaining / rate
                    hours = int(time_remaining // 3600)
                    minutes = int((time_remaining % 3600) // 60)
                    seconds = int(time_remaining % 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = "in the middle of calculating"
                
                print(f"  Indexed {total_papers:,}/{target_papers:,} papers | "
                      f"Rate: {rate:.1f} papers/sec | "
                      f"Time Remaining: {time_str} | "
                      f"Unique words: {len(lexicon):,}")
                
                last_report_time = current_time
    
    total_time = time.time() - start_time
    
    # analyze pos distribution
    pos_summary = defaultdict(int)
    for word_info in lexicon.values():
        for pos, count in word_info["pos_counts"].items():
            pos_summary[pos] += count
    
    print(f"\nIndex building completed in {total_time/60:.2f} minutes")
    print(f"Final count: {total_papers:,} papers indexed")
    print(f"POS tag distribution:")
    for pos, count in sorted(pos_summary.items(), key=lambda x: x[1], reverse=True): #asceding order sort to show the count of pos tagged terms 
        percentage = (count / sum(pos_summary.values())) * 100
        print(f"  {pos}: {count:,} ({percentage:.1f}%)")
    
    return lexicon, forward_index, inverted_index, backward_index
def generate_indexes_parallel(target_papers=50000, batch_size=100, num_workers=None, memory_safe=True):
    """
    Generate indexes using parallel processing
    Processes batches as they arrive from generator 
    """
    print("=" * 70)
    print("TRUE STREAMING PARALLEL INDEXING WITH SCIENTIFIC POS TAGGING")
    print(f"Target: {target_papers:,} papers")
    print("=" * 70)
    
    # use fewer workers for scispaCy
    if num_workers is None:
        if memory_safe:
            # safe settings for scispaCy 
            num_workers = min(4, max(1, cpu_count() // 3))  # max 4 workers / can do 11 but that would take too much memory and might crash
        else:
            num_workers = max(1, cpu_count() - 1)
    
    print(f"Configuration:")
    print(f"  Target papers: {target_papers:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Worker processes: {num_workers} (optimized for scispaCy)")
    print(f"  Model: en_core_sci_sm (scientific terminology)")
    print(f"  Processing mode: True streaming (no pre-collection)")
    print(f"  Memory safe: {'Yes' if memory_safe else 'No'}")
    print("=" * 70)
    
    # Get batch generator (we're not collecting or storing batches anywhere, just getting the batch stream)
    print(f"\nStep 1: starting paper stream...")
    batch_generator = get_paper_batches(batch_size=batch_size, max_papers=target_papers)
    
    print(f"\nStep 2: streaming processing with {num_workers} workers...")
    print("using en_core_sci_sm for scientific terminology")
    start_time = time.time()
    last_report_time = start_time
    
    batch_results = []
    completed_batches = 0
    total_papers_processed = 0
    
    # process chunks as they arrive from generator
    chunk_size = 20  # Process 20 batches at a time // yield generates paper stream -> stream collected as a batch -> batches collected into chunks -> chunks processed in parallel
    
    with Pool(processes=num_workers, initializer=init_worker_nlp) as pool:
        current_chunk = []
        
        for batch in batch_generator:  # streaming from generator, not collecting 
            current_chunk.append(batch)
            
            # when we have enough batches, process this chunk
            if len(current_chunk) >= chunk_size:
                # process the current chunk
                try:
                    chunk_results = list(pool.imap(process_paper_batch, current_chunk, chunksize=1))
                    
                    # add results
                    for result in chunk_results:
                        batch_results.append(result)
                        completed_batches += 1
                        total_papers_processed += len(result)
                        
                        # progress updates
                        current_time = time.time()
                        if completed_batches % 5 == 0 or (current_time - last_report_time) > 30:
                            elapsed = current_time - start_time
                            rate = total_papers_processed / elapsed
                            
                            # time remaining calculation
                            if rate > 0:
                                papers_remaining = target_papers - total_papers_processed
                                time_remaining = papers_remaining / rate
                                hours = int(time_remaining // 3600)
                                minutes = int((time_remaining % 3600) // 60)
                                seconds = int(time_remaining % 60)
                                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                                
                                # estimate completion time
                                completion_time = time.strftime("%H:%M:%S", 
                                    time.localtime(current_time + time_remaining))
                            else:
                                time_str = "calculating"
                                completion_time = "unknown"
                            
                            print(f"  Processed {completed_batches} batches | "
                                  f"{total_papers_processed:,}/{target_papers:,} papers | "
                                  f"Rate: {rate:.1f} papers/sec | "
                                  f"Remaining: {time_str} | "
                                  f"ETA: {completion_time}")
                            
                            last_report_time = current_time
                
                except Exception as e:
                    print(f"  Warning: Error processing chunk: {str(e)[:100]}")
                
                finally:
                    # Clear current chunk and memory
                    current_chunk = []
                    gc.collect()
        
        # process any remaining batches in the final chunk
        if current_chunk:
            try:
                chunk_results = list(pool.imap(process_paper_batch, current_chunk, chunksize=1))
                batch_results.extend(chunk_results)
                completed_batches += len(current_chunk)
                total_papers_processed += sum(len(r) for r in chunk_results)
            except Exception as e:
                print(f"  Warning: Error processing final chunk: {str(e)[:100]}")
    
    parallel_time = time.time() - start_time
    print(f"\nStreaming processing completed in {parallel_time/60:.2f} minutes")
    print(f"Total batches processed: {completed_batches}")
    print(f"Total papers processed: {total_papers_processed:,}")
    
    # Build indexes from results
    print(f"\nStep 3: Building indexes with POS information...")
    lexicon, forward_index, inverted_index, backward_index = build_indexes_from_batch_results(
        batch_results, target_papers=target_papers
    )
    
    total_time = time.time() - start_time
    total_papers_indexed = len(forward_index)
    
    print("\n" + "=" * 70)
    print("STREAMING INDEXING COMPLETE")
    print("=" * 70)
    print(f"Target papers: {target_papers:,}")
    print(f"Batches processed: {completed_batches}")
    print(f"Papers processed: {total_papers_processed:,}")
    print(f"Papers indexed: {total_papers_indexed:,}")
    
    if total_papers_indexed > 0:
        print(f"\nSuccess rates:")
        print(f"  Processing: {total_papers_processed/target_papers*100:.1f}%")
        print(f"  Final: {total_papers_indexed/target_papers*100:.1f}%")
    
    print(f"\nPerformance:")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Average rate: {total_papers_indexed/total_time:.1f} papers/sec")
    print(f"  Peak memory efficiency: True streaming (no batch pre-collection)")
    print(f"  Unique words: {len(lexicon):,}")
    print(f"  Index sizes:")
    print(f"    - Forward index: {len(forward_index):,} documents")
    print(f"    - Inverted index: {len(inverted_index):,} terms")
    print(f"    - Backward index: {len(backward_index):,} documents")
    print(f"    - Lexicon: {len(lexicon):,} entries with POS info")
    
    # check if we met the target
    if total_papers_indexed >= target_papers * 0.95:  
        print(f"\nIndexing successful: Achieved {total_papers_indexed/target_papers*100:.1f}% of target")
    elif total_papers_indexed >= target_papers * 0.8: 
        print(f"\nOnly achieved {total_papers_indexed/target_papers*100:.1f}% of target")
    else:
        print(f"\nOnly achieved {total_papers_indexed/target_papers*100:.1f}% of target")
    
    print("=" * 70)
    
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
    BATCH_SIZE = 100
    USE_PARALLEL = True
    MEMORY_SAFE = True  #enable memory optimizations
    NUM_WORKERS = 4    #fixed at 4 workers for scispaCy
    
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