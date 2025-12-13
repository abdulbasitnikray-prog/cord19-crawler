import json as js 
import pickle 
import struct 
import os 
import time 
import math
import re
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from singlewordSearch import (
    search_word, get_lemma_for_word, resolve_document_ids,
    doc_manager, DOC_MAP_PATH, barrel_lookup
)

# ============================================================================
# optimised preprocessing functions found here
# ============================================================================

# compile regex patterns once for performance (similar to stopword filtering, doesnt need compilation each time)
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
    'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 
    'might', 'must', 'about', 'above', 'after', 'before', 'between',
    'from', 'into', 'through', 'during', 'since', 'under', 'over',
    'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'just', 'now'
}

#cache for word expansions
_word_expansion_cache = {}

def expand_word_with_lemmas(word: str) -> List[str]:
    """Fast word expansion with caching"""
    if word in _word_expansion_cache:
        return _word_expansion_cache[word]
    
    lemmas = get_lemma_for_word(word, return_all_variations=True)
    expanded = [word.lower()]
    
    # add all lemma forms (e.g., plural, tense)
    for lemma in lemmas:
        lemma_lower = lemma.lower()
        if lemma_lower not in expanded:
            expanded.append(lemma_lower)
    
    # also add stripped version
    stripped = word.lower().strip('.,!?;:"\'()[]{}')
    if stripped and stripped not in expanded:
        expanded.append(stripped)
    
    result = list(set(expanded))
    _word_expansion_cache[word] = result
    return result

def preprocess_query(query: str) -> List[List[str]]:
    if not query:
        return []

    query = query.lower().strip()
    words = re.findall(r'\b[a-z0-9]{2,}\b', query)
    
    if not words:
        return []
    
    #filter stopwords
    filtered_words = [word for word in words if word not in STOPWORDS]
    
    #expand words
    expanded_words = []
    for word in filtered_words:
        word_forms = expand_word_with_lemmas(word)
        if word_forms:
            expanded_words.append(word_forms)
    
    return expanded_words

def search_word_with_variants(word_forms: List[str]) -> Dict[str, Tuple[List[str], Optional[List[int]]]]:
    #optimized parallel search for word variants
    results = {}
    
    # filter out words that don't exist in any barrel using trie
    existing_words = []
    for word_form in word_forms:
        barrel_ids = barrel_lookup.get_barrels_for_word(word_form)
        if barrel_ids:
            existing_words.append(word_form)
    
    if not existing_words:
        return {}
    
    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(existing_words))) as executor:
        future_to_word = {
            executor.submit(search_word, word_form, True): word_form 
            for word_form in existing_words
        }
        
        # Collect results as they complete (save memory by not storing futures)
        for future in concurrent.futures.as_completed(future_to_word):
            word_form = future_to_word[future]
            try:
                doc_ids, frequencies = future.result()
                if doc_ids:
                    str_doc_ids = resolve_document_ids(doc_ids)
                    results[word_form] = (str_doc_ids, frequencies)
            except Exception:
                continue
    
    return results

def search_words_batch(word_groups: List[List[str]]) -> List[Dict[str, Tuple[List[str], Optional[List[int]]]]]:
    #Batch search for multiple word groups in parallel

    all_word_results = [None] * len(word_groups)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(word_groups))) as executor:

        future_to_idx = {
            executor.submit(search_word_with_variants, word_forms): idx
            for idx, word_forms in enumerate(word_groups)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                word_results = future.result()
                if word_results:
                    all_word_results[idx] = word_results
            except Exception:
                all_word_results[idx] = {}
    
    return all_word_results

def combine_word_results_fast(all_word_results: List[Dict[str, Tuple[List[str], Optional[List[int]]]]]) -> Dict[str, float]:
    if not all_word_results:
        return {}
    # Load total document count from mapping
    try:
        with open(DOC_MAP_PATH, 'rb') as f:
            mapping_data = pickle.load(f)
        total_docs = len(mapping_data.get("int_to_str", {}))
    except:
        total_docs = 50000  # Reasonable default
    
    # Phase 1: Collect document statistics
    doc_word_freqs = defaultdict(lambda: defaultdict(int))
    word_doc_counts = defaultdict(int)
    
    for word_idx, word_result in enumerate(all_word_results):
        if not word_result:
            continue
            
        # Track best frequency for each document
        doc_freqs_for_word = defaultdict(int)
        
        for _, (doc_ids, frequencies) in word_result.items():
            if frequencies and len(frequencies) == len(doc_ids):
                # Use frequencies if available
                for doc_id, freq in zip(doc_ids, frequencies):
                    if freq > doc_freqs_for_word[doc_id]:
                        doc_freqs_for_word[doc_id] = freq
            else:
                for doc_id in doc_ids:
                    if doc_freqs_for_word[doc_id] == 0:
                        doc_freqs_for_word[doc_id] = 1
        
        # global statistics being built
        for doc_id, freq in doc_freqs_for_word.items():
            doc_word_freqs[doc_id][word_idx] = freq
        
        word_doc_counts[word_idx] = len(doc_freqs_for_word)
    
    # Phase 2: TF-IDF Scoring (basically tells you how relevant a document is for the query)
    doc_scores = defaultdict(float)
    total_words = len(all_word_results)
    
    # Pre-calculate IDF for each word index (pre calc necessary to speed up otherwise recalculated for each doc)
    word_idf = {}
    for word_idx in range(total_words):
        doc_count = word_doc_counts[word_idx]
        if doc_count > 0:
            # optimisation using smoothed IDF
            word_idf[word_idx] = math.log((total_docs + 1) / (doc_count + 1)) + 1.0
        else:
            word_idf[word_idx] = 1.0
    
    # Calculate scores only for documents that appear in results (Saves time)
    for doc_id, word_freqs in doc_word_freqs.items():
        score = 0.0
        word_count = len(word_freqs)
        
        if word_count > 0:
            for word_idx, freq in word_freqs.items():
                # Fast TF: log(1 + freq)
                tf = 1.0 + math.log(freq) if freq > 1 else 1.0
                score += tf * word_idf[word_idx]
            
            # Bonus for having more query words (encourages comprehensive matches)
            if word_count > 1:
                score *= (1.0 + 0.1 * (word_count - 1))
            
            doc_scores[doc_id] = score
    
    return dict(doc_scores)

def multi_word_search(query: str, max_results: int = 20) -> List[Tuple[str, float]]:

    start_time = time.time()
    
    # Fast preprocessing
    expanded_words = preprocess_query(query)
    
    if not expanded_words:
        print(f"No valid search terms")
        return []
    
    print(f"\n-- MULTI-WORD SEARCH --: '{query}'")
    print(f"Search terms: {[words[0] for words in expanded_words]}")
    
    #Batch search for all word groups in parallel
    all_word_results = search_words_batch(expanded_words)
    
    #Filter out empty results
    valid_word_results = [wr for wr in all_word_results if wr]
    
    if not valid_word_results:
        print(f"No documents found")
        return []
    
    # Combine results
    combined_results = combine_word_results_fast(valid_word_results)
    
    if not combined_results:
        print(f"No matching documents")
        return []
    
    # Fast top-K selection with heapq (this is an optimisation which avoids full sort if not needed)
    if max_results < len(combined_results):
        top_items = heapq.nlargest(max_results, combined_results.items(), key=lambda x: x[1])
        final_results = top_items
    else:
        final_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"Found {len(combined_results)} documents in {elapsed_time:.3f}s")
    
    # Performance feedback (to attach in interactive mode later)
    word_count = len(expanded_words)
    if word_count == 1 and elapsed_time > 0.5:
        print(f"Single word query: {elapsed_time:.3f}s (target: <0.5s)")
    elif word_count == 5 and elapsed_time > 1.5:
        print(f"5-word query: {elapsed_time:.3f}s (target: <1.5s)")
    
    return final_results

# ============================================================================
#display functions (Almost same as singlewordSearch.py)
# ============================================================================

def display_multi_word_results(query: str, results: List[Tuple[str, float]], max_display: int = 10):

    if not results:
        print(f"\n📭 No documents found")
        return
    
    total_docs = len(results)
    display_count = min(max_display, total_docs)
    
    print(f"\n{'='*80}")
    print(f"TOP {display_count} RESULTS (out of {total_docs})")
    print(f"{'='*80}\n")
    
    #Batch fetch titles for display
    display_items = results[:display_count]
    titles = {}
    
    for doc_id, _ in display_items:
        titles[doc_id] = doc_manager.get_document_title(doc_id)
    
    for i, (doc_id, score) in enumerate(display_items, 1):
        title = titles[doc_id]
        
        #Truncate long titles
        if len(title) > 70:
            title = title[:67] + "..."
        
        print(f"{i:2d}. {title}")
        print(f"    ID: {doc_id} | Score: {score:.2f}")
        
        # Optional: Show preview for top 3 results
        if i <= 3:
            # Quick context from title
            if "covid" in query.lower() or "corona" in query.lower():
                # Highlight COVID-related content
                if any(term in title.lower() for term in ["covid", "corona", "sars", "pandemic"]):
                    print(f"COVID-related research")
            print()
    
    # Summary
    if total_docs > display_count:
        print(f"... and {total_docs - display_count} more documents")
    print(f"{'='*80}")

# ============================================================================
# perforamnce test functions
# ============================================================================

def performance_test():

    print("\n" + "="*80)
    print("MWS PERFORMANCE TEST")
    print("="*80)
    
    # Ensure trie is loaded
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    test_queries = [
        ("covid", 1),
        ("covid vaccine", 2),
        ("coronavirus treatment pandemic", 3),
        ("sars cov 2 transmission mask", 4),
        ("ventilator icu patients oxygen therapy", 5),
        ("covid 19 pandemic response and treatment strategies", 6)
    ]
    
    print("\n--Running performance tests--\n")
    
    all_passed = True
    
    for query, word_count in test_queries:
        print(f"Testing: '{query}' ({word_count} words)")
    
        _ = multi_word_search("covid", max_results=1)

        start_time = time.time()
        results = multi_word_search(query, max_results=10)
        elapsed = time.time() - start_time
        
        found = len(results)
        
        # Check performance constraints
        if word_count == 1:
            if elapsed > 0.5:
                status = "didn't pass"
                all_passed = False
            elif elapsed > 0.4:
                status = "suboptimal"
            else:
                status = "good"
        elif word_count == 5:
            if elapsed > 1.5:
                status = "didn't pass"
                all_passed = False
            elif elapsed > 1.2:
                status = "suboptimal"
            else:
                status = "good"
        else:
            if elapsed > 2.0:
                status = "didn't pass"
            else:
                status = "good"
        
        print(f"{status} {elapsed:.3f}s, found {found} docs")
        
        if results:
            # Show top result
            doc_id, score = results[0]
            title = doc_manager.get_document_title(doc_id)
            if len(title) > 50:
                title = title[:47] + "..."
            print(f"   Top: {title} ({score:.2f})")
        
        print()
    
    print("="*80)
    if all_passed:
        print("system passed all performance targets!")
    else:
        print("some tests did not meet performance targets.")
    print("="*80)

# ============================================================================
# interactive functions
# ============================================================================

def interactive_multi_word_search():
    print("\n" + "="*80)
    print("MWS INTERFACE")
    print("="*80)
    print("Enter multi-word queries like:")
    print("• covid vaccine")
    print("• coronavirus treatment pandemic")
    print("• sars cov 2 transmission")
    print("="*80)
    
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    while True:
        try:
            query = input("\nEnter search query (or 'quit'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("ciao!")
                break
            
            if not query:
                continue
            
            #measure performance
            start_time = time.time()
            results = multi_word_search(query, max_results=20)
            elapsed = time.time() - start_time
            
            if results:
                display_multi_word_results(query, results, max_display=10)
                print(f"\nSearch completed in {elapsed:.3f}s")
                
                #Performance feedback
                word_count = len(preprocess_query(query))
                if word_count == 1 and elapsed > 0.5:
                    print(f"Single word should be < 0.5s")
                elif word_count == 5 and elapsed > 1.5:
                    print(f"5 words should be < 1.5s")
            else:
                print(f"\nNo results found in {elapsed:.3f}s")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)[:100]}")

# ============================================================================
# main execution
# ============================================================================

def test_multi_word_search():
    print("\n" + "=" * 80)
    print("MWS basic functionality test")
    print("=" * 80)

    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    test_queries = [
        "covid",
        "covid vaccine",
        "coronavirus treatment",
        "sars cov 2",
        "mask social distancing pandemic"
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Testing: '{query}'")
        
        results = multi_word_search(query, max_results=5)
        
        if results:
            display_multi_word_results(query, results, max_display=3)
        else:
            print(f"No results found")
    
    print(f"\n{'=' * 80}")
    print("Testing finished")
    print("=" * 80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CORD-19 optimised MWS Module")
    print("="*80)
    
    print("\n1. Performance test")
    print("2. Basic search test")
    print("3. Interactive search")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        performance_test()
    elif choice == "2":
        test_multi_word_search()
    elif choice == "3":
        interactive_multi_word_search()
    else:
        print("Goodbye!")