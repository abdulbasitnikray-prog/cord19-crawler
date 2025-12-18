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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARRELS_DIR = os.path.join(BASE_DIR, "data", "barrels")
COMPRESSED_BARRELS_DIR = os.path.join(BASE_DIR, "data", "compressed_barrels")
BARREL_MAP_PATH = os.path.join(BARRELS_DIR, "barrel_mappings.json")

# Global Cache for total docs
_total_docs_cache = None

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 
    'may', 'might', 'must', 'about', 'above', 'after', 'before', 'between', 'from', 'into', 'through', 'during', 'since', 
    'under', 'over', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'just', 'now'
}

_word_expansion_cache = {}

def expand_word_with_lemmas(word: str) -> List[str]:
    if word in _word_expansion_cache: return _word_expansion_cache[word]
    lemmas = get_lemma_for_word(word, return_all_variations=True)
    expanded = [word.lower()]
    for lemma in lemmas:
        if lemma.lower() not in expanded: expanded.append(lemma.lower())
    result = list(set(expanded))
    _word_expansion_cache[word] = result
    return result

def preprocess_query(query: str) -> List[List[str]]:
    if not query: return []
    query = query.lower().strip()
    words = re.findall(r'\b[a-z0-9]{2,}\b', query)
    if not words: return []
    filtered = [w for w in words if w not in STOPWORDS]
    return [expand_word_with_lemmas(w) for w in filtered if expand_word_with_lemmas(w)]

def search_word_with_variants(word_forms: List[str]) -> Dict[str, Tuple[List[str], Optional[List[int]]]]:
    results = {}
    existing = []
    for wf in word_forms:
        try:
            if barrel_lookup.get_barrels_for_word(wf): existing.append(wf)
        except: continue
    
    if not existing: existing = word_forms
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(existing) + 1)) as executor:
        future_to_word = {executor.submit(search_word, wf, True): wf for wf in existing}
        for future in concurrent.futures.as_completed(future_to_word):
            wf = future_to_word[future]
            try:
                ids, freqs = future.result()
                if ids:
                    s_ids = resolve_document_ids(ids)
                    if s_ids: results[wf] = (s_ids, freqs)
            except: continue
    return results

def search_words_batch(word_groups: List[List[str]]) -> List[Dict]:
    all_res = [None] * len(word_groups)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(word_groups))) as executor:
        future_to_idx = {executor.submit(search_word_with_variants, wg): i for i, wg in enumerate(word_groups)}
        for future in concurrent.futures.as_completed(future_to_idx):
            all_res[future_to_idx[future]] = future.result() or {}
    return all_res

def combine_word_results_fast(all_word_results: List[Dict]) -> Dict[str, float]:
    valid = [wr for wr in all_word_results if wr]
    if not valid: return {}
    
    # --- OPTIMIZATION: Cache Total Docs to prevent IO lag ---
    global _total_docs_cache
    if _total_docs_cache is None:
        try:
            with open(DOC_MAP_PATH, 'rb') as f:
                _total_docs_cache = len(pickle.load(f).get("int_to_str", {}))
        except: _total_docs_cache = 50000
    total_docs = _total_docs_cache
    # --------------------------------------------------------
    
    doc_word_freqs = defaultdict(lambda: defaultdict(int))
    word_doc_counts = defaultdict(int)
    
    for idx, wr in enumerate(valid):
        doc_freqs_for_word = defaultdict(int)
        for _, (ids, freqs) in wr.items():
            if freqs and len(freqs) == len(ids):
                for doc_id, f in zip(ids, freqs):
                    if f > doc_freqs_for_word[doc_id]: doc_freqs_for_word[doc_id] = f
            else:
                for doc_id in ids:
                    if doc_freqs_for_word[doc_id] == 0: doc_freqs_for_word[doc_id] = 1
        
        for doc_id, f in doc_freqs_for_word.items():
            doc_word_freqs[doc_id][idx] = f
        word_doc_counts[idx] = len(doc_freqs_for_word)
        
    doc_scores = defaultdict(float)
    total_words = len(valid)
    word_idf = {}
    
    for i in range(total_words):
        dc = word_doc_counts[i]
        word_idf[i] = math.log((total_docs + 1) / (dc + 1)) + 1.0 if dc > 0 else 1.0
        
    for doc_id, wfs in doc_word_freqs.items():
        score = 0.0
        wc = len(wfs)
        if wc > 0:
            for idx, f in wfs.items():
                tf = 1.0 + math.log(f) if f > 1 else 1.0
                score += tf * word_idf[idx]
            if wc > 1: score *= (1.0 + 0.1 * (wc - 1))
            doc_scores[doc_id] = score
            
    return dict(doc_scores)

def multi_word_search(query: str, max_results: int = 20) -> Tuple[List[Tuple[str, float]], int]:
    start = time.time()
    expanded = preprocess_query(query)
    if not expanded: return [], 0
    
    print(f"\n-- MULTI-WORD SEARCH --: '{query}'")
    all_res = search_words_batch(expanded)
    valid = [wr for wr in all_res if wr]
    if not valid: return [], 0
    
    combined = combine_word_results_fast(valid)
    if not combined: return [], 0
    
    if max_results < len(combined):
        final = heapq.nlargest(max_results, combined.items(), key=lambda x: x[1])
    else:
        final = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
    elapsed = time.time() - start
    print(f"✅ Found {len(combined)} documents in {elapsed:.3f}s")
    return final, len(combined)
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
    print("[*] Loading barrel trie...")
    barrel_lookup.load_trie()
    print("[*] Loading document metadata...")
    doc_manager.load_metadata()
    print("[✓] Resources loaded\n")
    
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
    
        start_time = time.time()
        results, found = multi_word_search(query, max_results=10)
        elapsed = time.time() - start_time
        
        # Check performance constraints
        if word_count == 1:
            if elapsed > 0.5:
                status = "❌ didn't pass"
                all_passed = False
            elif elapsed > 0.4:
                status = "⚠️ suboptimal"
            else:
                status = "✅ good"
        elif word_count == 5:
            if elapsed > 1.5:
                status = "❌ didn't pass"
                all_passed = False
            elif elapsed > 1.2:
                status = "⚠️ suboptimal"
            else:
                status = "✅ good"
        else:
            if elapsed > 2.0:
                status = "❌ didn't pass"
                all_passed = False
            else:
                status = "✅ good"
        
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
        print("✅ system passed all performance targets!")
    else:
        print("⚠️ some tests did not meet performance targets.")
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
    
    print("[*] Loading barrel trie...")
    barrel_lookup.load_trie()
    print("[*] Loading document metadata...")
    doc_manager.load_metadata()
    print("[✓] Resources loaded\n")
    
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
            results, found = multi_word_search(query, max_results=20)
            elapsed = time.time() - start_time
            
            if results:
                display_multi_word_results(query, results, max_display=10)
                print(f"\nSearch completed in {elapsed:.3f}s")
                
                #Performance feedback
                word_count = len(preprocess_query(query))
                if word_count == 1 and elapsed > 0.5:
                    print(f"⚠️ Single word should be < 0.5s")
                elif word_count == 5 and elapsed > 1.5:
                    print(f"⚠️ 5 words should be < 1.5s")
            else:
                print(f"\n❌ No results found in {elapsed:.3f}s")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)[:100]}")

# ============================================================================
# main execution
# ============================================================================

def test_multi_word_search():
    print("\n" + "=" * 80)
    print("MWS basic functionality test")
    print("=" * 80)

    print("[*] Loading barrel trie...")
    barrel_lookup.load_trie()
    print("[*] Loading document metadata...")
    doc_manager.load_metadata()
    print("[✓] Resources loaded\n")
    
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
        
        results, found = multi_word_search(query, max_results=5)
        
        if results:
            display_multi_word_results(query, results, max_display=3)
        else:
            print(f"❌ No results found")
    
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