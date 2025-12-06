# index.py - Parallel indexing with accurate counting and time estimates
import json as js 
import os
import csv
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from crawler import get_paper_batches, process_paper_batch, init_worker_nlp

# Base path
BASE_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10"

def build_indexes_from_batch_results(batch_results, target_papers=50000):
    """
    Build all indexes from processed batch results
    """
    print(f"\n--- Building Indexes from Processed Results (Target: {target_papers:,} papers) ---")
    
    lexicon = {}           # word -> word_id
    forward_index = {}     # doc_id -> [word_id1, word_id2, ...]
    inverted_index = {}    # word_id -> {doc_id: frequency}
    backward_index = {}    # doc_id -> [word1, word2, ...]
    word_id_counter = 1
    
    total_papers = 0
    start_time = time.time()
    last_report_time = start_time
    
    # Process all batch results
    for batch_idx, batch in enumerate(batch_results):
        for paper_data in batch:
            if not paper_data or "tokens" not in paper_data:
                continue
                
            doc_id = paper_data["cord_uid"]
            tokens = paper_data["tokens"]
            
            # Update backward index
            backward_index[doc_id] = tokens
            
            # Update lexicon and forward index
            doc_word_ids = []
            word_freq = defaultdict(int)
            
            for word in tokens:
                # Add to lexicon if new
                if word not in lexicon:
                    lexicon[word] = word_id_counter
                    word_id_counter += 1
                
                word_id = lexicon[word]
                doc_word_ids.append(word_id)
                word_freq[word_id] += 1
            
            # Update forward index
            forward_index[doc_id] = doc_word_ids
            
            # Update inverted index
            for word_id, freq in word_freq.items():
                if word_id not in inverted_index:
                    inverted_index[word_id] = {}
                inverted_index[word_id][doc_id] = freq
            
            total_papers += 1
            
            # Progress reporting every 1000 papers or 30 seconds
            current_time = time.time()
            if total_papers % 1000 == 0 or (current_time - last_report_time) > 30:
                elapsed = current_time - start_time
                rate = total_papers / elapsed
                
                # Calculate time remaining
                papers_remaining = target_papers - total_papers
                if rate > 0:
                    time_remaining = papers_remaining / rate
                    hours = int(time_remaining // 3600)
                    minutes = int((time_remaining % 3600) // 60)
                    seconds = int(time_remaining % 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = "Calculating..."
                
                print(f"  Indexed {total_papers:,}/{target_papers:,} papers | "
                      f"Rate: {rate:.1f} papers/sec | "
                      f"Remaining: {time_str} | "
                      f"Unique words: {len(lexicon):,}")
                
                last_report_time = current_time
    
    total_time = time.time() - start_time
    print(f"\nIndex building completed in {total_time/60:.2f} minutes")
    print(f"Final count: {total_papers:,} papers indexed")
    
    return lexicon, forward_index, inverted_index, backward_index

def generate_indexes_parallel(target_papers=50000, batch_size=100, num_workers=None):
    """
    Generate indexes using parallel batch processing with accurate counting
    """
    print("=" * 70)
    print("PARALLEL INDEX GENERATION")
    print(f"Target: {target_papers:,} papers")
    print("=" * 70)
    
    # Configuration
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print(f"Configuration:")
    print(f"  Target papers: {target_papers:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Worker processes: {num_workers}")
    print(f"  Expected batches: ~{target_papers // batch_size}")
    print("=" * 70)
    
    # Create batches
    print(f"\nStep 1: Creating paper batches...")
    batch_generator = get_paper_batches(batch_size=batch_size, max_papers=target_papers)
    
    # Convert generator to list for parallel processing
    print("Collecting batches...")
    all_batches = []
    batch_count = 0
    total_papers_collected = 0
    batch_collect_start = time.time()
    
    for batch in batch_generator:
        all_batches.append(batch)
        batch_count += 1
        total_papers_collected += len(batch)
        
        if batch_count % 10 == 0:
            elapsed = time.time() - batch_collect_start
            rate = total_papers_collected / elapsed
            papers_remaining = target_papers - total_papers_collected
            
            if rate > 0:
                time_remaining = papers_remaining / rate
                hours = int(time_remaining // 3600)
                minutes = int((time_remaining % 3600) // 60)
                time_str = f"{hours:02d}:{minutes:02d}"
            else:
                time_str = "Calculating..."
            
            print(f"  Collected {batch_count} batches | "
                  f"{total_papers_collected:,}/{target_papers:,} papers | "
                  f"Rate: {rate:.1f} papers/sec | "
                  f"Remaining: {time_str}")
    
    print(f"Total batches collected: {batch_count}")
    print(f"Total papers collected: {total_papers_collected:,}")
    
    if total_papers_collected < target_papers * 0.9:  # Less than 90% of target
        print(f"⚠ Warning: Only collected {total_papers_collected:,} papers "
              f"({total_papers_collected/target_papers*100:.1f}% of target)")
    
    # Process batches in parallel
    print(f"\nStep 2: Processing batches with {num_workers} workers...")
    start_time = time.time()
    last_report_time = start_time
    
    with Pool(processes=num_workers, initializer=init_worker_nlp) as pool:
        # Use imap_unordered for better performance
        batch_results = []
        completed = 0
        total_papers_processed = 0
        
        for result in pool.imap_unordered(process_paper_batch, all_batches, chunksize=2):
            batch_results.append(result)
            completed += 1
            total_papers_processed += len(result)
            
            current_time = time.time()
            if completed % 5 == 0 or (current_time - last_report_time) > 30:
                elapsed = current_time - start_time
                rate = total_papers_processed / elapsed
                batches_remaining = len(all_batches) - completed
                
                # Calculate time remaining
                if rate > 0:
                    papers_remaining = target_papers - total_papers_processed
                    time_remaining = papers_remaining / rate
                    hours = int(time_remaining // 3600)
                    minutes = int((time_remaining % 3600) // 60)
                    seconds = int(time_remaining % 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # Estimate completion time
                    completion_time = time.strftime("%H:%M:%S", 
                        time.localtime(current_time + time_remaining))
                else:
                    time_str = "Calculating..."
                    completion_time = "Unknown"
                
                print(f"  Batch {completed}/{len(all_batches)} | "
                      f"{total_papers_processed:,}/{target_papers:,} papers | "
                      f"Rate: {rate:.1f} papers/sec | "
                      f"Remaining: {time_str} | "
                      f"ETA: {completion_time}")
                
                last_report_time = current_time
    
    parallel_time = time.time() - start_time
    print(f"\nParallel processing completed in {parallel_time/60:.2f} minutes")
    print(f"Total papers processed: {total_papers_processed:,}")
    
    # Build indexes from results
    print(f"\nStep 3: Building indexes from processed results...")
    lexicon, forward_index, inverted_index, backward_index = build_indexes_from_batch_results(
        batch_results, target_papers=target_papers
    )
    
    total_time = time.time() - start_time
    total_papers_indexed = len(forward_index)
    
    print("\n" + "=" * 70)
    print("PARALLEL INDEXING COMPLETE")
    print("=" * 70)
    print(f"Target papers: {target_papers:,}")
    print(f"Papers collected: {total_papers_collected:,}")
    print(f"Papers processed: {total_papers_processed:,}")
    print(f"Papers indexed: {total_papers_indexed:,}")
    
    if total_papers_indexed > 0:
        print(f"\nSuccess rates:")
        print(f"  Collection: {total_papers_collected/target_papers*100:.1f}%")
        print(f"  Processing: {total_papers_processed/total_papers_collected*100:.1f}%")
        print(f"  Final: {total_papers_indexed/target_papers*100:.1f}%")
    
    print(f"\nPerformance:")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Average rate: {total_papers_indexed/total_time:.1f} papers/sec")
    print(f"  Unique words: {len(lexicon):,}")
    print(f"  Index sizes:")
    print(f"    - Forward index: {len(forward_index):,} documents")
    print(f"    - Inverted index: {len(inverted_index):,} terms")
    print(f"    - Backward index: {len(backward_index):,} documents")
    
    # Check if we met the target
    if total_papers_indexed >= target_papers * 0.95:  # 95% of target
        print(f"\n✅ Success: Achieved {total_papers_indexed/target_papers*100:.1f}% of target")
    elif total_papers_indexed >= target_papers * 0.8:  # 80% of target
        print(f"\n⚠ Warning: Only achieved {total_papers_indexed/target_papers*100:.1f}% of target")
    else:
        print(f"\n❌ Issue: Only achieved {total_papers_indexed/target_papers*100:.1f}% of target")
    
    print("=" * 70)
    
    return lexicon, forward_index, inverted_index, backward_index

def save_index_files(lexicon, forward_index, inverted_index, backward_index, output_dir="indexes"):
    """
    Save all index files to disk with statistics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        total_papers = len(forward_index)
        total_words = len(lexicon)
        
        print(f"\nSaving index files to '{output_dir}'...")
        print(f"Statistics: {total_papers:,} papers, {total_words:,} unique words")
        
        # Save Lexicon
        lexicon_path = os.path.join(output_dir, "lexicon.json")
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            js.dump(lexicon, f, indent=None)
        print(f"✓ lexicon.json: {len(lexicon):,} words")

        # Save Forward Index
        forward_path = os.path.join(output_dir, "forward_index.json")
        with open(forward_path, 'w', encoding='utf-8') as f:
            js.dump(forward_index, f, indent=None)
        print(f"✓ forward_index.json: {len(forward_index):,} documents")
        
        # Save Inverted Index
        inverted_path = os.path.join(output_dir, "inverted_index.json")
        with open(inverted_path, 'w', encoding='utf-8') as f:
            js.dump(inverted_index, f, indent=None)
        print(f"✓ inverted_index.json: {len(inverted_index):,} terms")
        
        # Save Backward Index
        backward_path = os.path.join(output_dir, "backward_index.json")
        with open(backward_path, 'w', encoding='utf-8') as f:
            js.dump(backward_index, f, indent=None)
        print(f"✓ backward_index.json: {len(backward_index):,} documents")
        
        # Save detailed statistics
        stats = {
            "total_papers_indexed": total_papers,
            "total_unique_words": total_words,
            "inverted_index_terms": len(inverted_index),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_minutes": round(time.time() - start_time) // 60,
            "papers_per_second": round(total_papers / (time.time() - start_time), 1),
            "index_sizes_bytes": {
                "lexicon": os.path.getsize(lexicon_path),
                "forward_index": os.path.getsize(forward_path),
                "inverted_index": os.path.getsize(inverted_path),
                "backward_index": os.path.getsize(backward_path)
            }
        }
        
        stats_path = os.path.join(output_dir, "index_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            js.dump(stats, f, indent=2)
        print(f"✓ index_statistics.json")
        
        print(f"\n✅ All index files saved to {os.path.abspath(output_dir)}")
            
    except Exception as e:
        print(f"Error saving files: {e}")

def main():
    """
    Main function
    """
    print("=" * 70)
    print("CORD-19 INDEX GENERATION")
    print("=" * 70)
    
    # Configuration
    TARGET_PAPERS = 50000
    BATCH_SIZE = 100
    USE_PARALLEL = True
    
    print(f"Target: Process {TARGET_PAPERS:,} papers")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mode: {'Parallel' if USE_PARALLEL else 'Single-threaded'}")
    print("=" * 70)
    
    global start_time
    start_time = time.time()
    
    # Generate indexes
    if USE_PARALLEL:
        lexicon, forward_index, inverted_index, backward_index = generate_indexes_parallel(
            target_papers=TARGET_PAPERS, 
            batch_size=BATCH_SIZE
        )
    else:
        # Single-threaded version would go here
        print("Single-threaded mode not implemented in this version")
        return
    
    # Save indexes
    if lexicon:  # Check if indexing was successful
        save_index_files(lexicon, forward_index, inverted_index, backward_index)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()