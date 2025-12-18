import time
import re
import os
import pandas as pd
from typing import List, Tuple, Dict, Optional
from singlewordSearch import search_word, resolve_document_ids, doc_manager, verify_paths, barrel_lookup
from multiwordSearch import multi_word_search, display_multi_word_results, preprocess_query
from ranking import PaperRanker, create_ranker

# Global ranker instance
ranker = None
citation_builder = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_ranking_system():
    """Initialize the ranking system with metadata and citation graph."""
    global ranker, citation_builder
    
    if ranker is not None:
        return ranker
    
    print("\nInitializing ranking system...")
    
    try:
        # Load metadata
        METADATA_PATH = "C:/Users/acer/Downloads/cord-19_2020-04-10/2020-04-10/metadata.csv"
        CITATION_GRAPH_PATH = "citation_graph.pkl"
        
        if not os.path.exists(METADATA_PATH):
            print(f"Metadata file not found at {METADATA_PATH}")
            return None
        
        # Load citation graph builder
        from citation_graph import build_citation_graph
        citation_builder = build_citation_graph(METADATA_PATH, CITATION_GRAPH_PATH)
        
        # Load metadata
        metadata_df = pd.read_csv(METADATA_PATH)
        
        # Create ranker
        ranker = PaperRanker(citation_builder, metadata_df)
        print("Ranking system initialized successfully")
        
        return ranker
        
    except Exception as e:
        print(f"Error initializing ranking system: {e}")
        return None

def get_paper_text_for_ranking(paper_ids: List[str], limit: int = 10) -> Dict[str, Dict]:
    """
    Extract text for papers to use in ranking.
    Only extracts for top papers to save time and memory.
    """
    paper_texts = {}
    
    # Process only top papers to minimize memory
    for paper_id in paper_ids[:limit]:
        text = doc_manager.get_document_text(paper_id)
        if text:
            # Simple text segmentation for ranking with limited size
            lines = text.split('\n', 100)  # Limit split to first 100
            title = lines[0][:500].lower() if lines else ""  # Limit title length
            abstract = " ".join(lines[1:5])[:1000].lower() if len(lines) > 1 else ""  # Limit abstract
            body = " ".join(lines[5:50])[:5000].lower() if len(lines) > 5 else ""  # Reduced from 100 to 50 lines
            
            paper_texts[paper_id] = {
                'title': title,
                'abstract': abstract,
                'body': body
            }
    
    return paper_texts

# ============================================================================
# ENHANCED SEARCH FUNCTIONS WITH RANKING
# ============================================================================

def smart_search_with_ranking(query: str, max_results: int = 20) -> List[Tuple[str, float]]:
    """
    Smart search with advanced ranking.
    """
    start_time = time.time()
    
    query = query.strip()
    if not query:
        return []
    
    # Get initial search results
    words = preprocess_query(query)
    
    if not words:
        print(f"No valid search terms in query")
        return []
    
    if len(words) == 1:
        # Single word search
        search_term = words[0][0]
        print(f"\nSINGLE WORD SEARCH: '{search_term}'")
        doc_ids, frequencies = search_word(search_term)
    else:
        # Multi-word search
        print(f"\nMULTI-WORD SEARCH: '{query}'")
        initial_results = multi_word_search(query, max_results * 2)
        if not initial_results:
            return []
        doc_ids = [doc_id for doc_id, _ in initial_results]
        frequencies = None
    
    if not doc_ids:
        elapsed = time.time() - start_time
        print(f"No documents found in {elapsed:.3f}s")
        return []
    
    # Convert to string IDs
    str_doc_ids = resolve_document_ids(doc_ids)
    
    # Initialize ranking system if not already done
    global ranker
    if ranker is None:
        ranker = initialize_ranking_system()
    
    # Apply advanced ranking if available
    if ranker:
        print(f"Applying advanced ranking to {len(str_doc_ids)} documents...")
        
        # Get text for top papers for better ranking
        paper_texts = get_paper_text_for_ranking(str_doc_ids, limit=min(50, len(str_doc_ids)))
        
        # Rank papers
        ranked_results = ranker.rank_papers(str_doc_ids, query, paper_texts)
        
        # Limit to max_results
        final_results = ranked_results[:max_results]
        
        # If we have frequencies for single-word search, blend with ranking
        if len(words) == 1 and frequencies and len(frequencies) == len(doc_ids):
            # Create frequency map
            freq_map = {str_id: freq for str_id, freq in zip(str_doc_ids, frequencies)}
            
            # Adjust scores based on frequency (25% weight to frequency, 75% to ranking)
            adjusted_results = []
            for doc_id, rank_score in final_results:
                freq = freq_map.get(doc_id, 0)
                freq_score = min(freq * 0.1, 1.0)  # Normalize frequency to 0-1
                final_score = (rank_score * 0.75) + (freq_score * 0.25)
                adjusted_results.append((doc_id, final_score))
            
            final_results = adjusted_results
    else:
        # Fallback: simple ranking by frequency/position
        print("Using simple ranking (advanced ranking not available)")
        if frequencies and len(frequencies) == len(str_doc_ids):
            # Sort by frequency
            sorted_items = sorted(zip(str_doc_ids, frequencies), key=lambda x: x[1], reverse=True)
            final_results = [(doc_id, min(freq * 0.5, 10.0)) for doc_id, freq in sorted_items[:max_results]]
        else:
            # Simple position-based ranking
            final_results = [(doc_id, 10.0 - (i * 0.1)) for i, doc_id in enumerate(str_doc_ids[:max_results])]
    
    elapsed = time.time() - start_time
    
    if ranker:
        print(f"Found and ranked {len(final_results)} documents in {elapsed:.3f}s")
    else:
        print(f"Found {len(final_results)} documents in {elapsed:.3f}s")
    
    return final_results

def display_ranked_results(query: str, results: List[Tuple[str, float]], max_display: int = 10):
    """
    Display search results with ranking information.
    """
    if not results:
        print(f"\nNo documents found for query: '{query}'")
        return
    
    # Determine query type
    words = preprocess_query(query)
    is_single_word = len(words) == 1
    
    total_docs = len(results)
    display_count = min(max_display, total_docs)
    
    # Display header
    print(f"\n{'='*80}")
    if is_single_word:
        print(f"SINGLE WORD RESULTS: '{query.upper()}'")
    else:
        print(f"MULTI-WORD RESULTS: '{query.upper()}'")
    
    if ranker:
        print(f"Advanced ranking applied")
    
    print(f"{'='*80}")
    print(f"Showing {display_count} of {total_docs} documents\n")
    
    # Batch fetch titles
    display_items = results[:display_count]
    titles = {}
    
    for doc_id, _ in display_items:
        titles[doc_id] = doc_manager.get_document_title(doc_id)
    
    # Display results
    for i, (doc_id, score) in enumerate(display_items, 1):
        title = titles[doc_id]
        
        # Truncate long titles
        if len(title) > 70:
            title = title[:67] + "..."
        
        # Score visualization
        star_count = min(int(score * 10), 5)  # Scale to 5 stars
        stars = "★" * star_count + "☆" * (5 - star_count)
        
        # Get additional info if available
        metadata_info = ""
        if ranker and doc_id in ranker.paper_metadata:
            metadata = ranker.paper_metadata[doc_id]
            year = ""
            publish_time = str(metadata.get('publish_time', ''))
            year_match = re.search(r'(\d{4})', publish_time)
            if year_match:
                year = f" | {year_match.group(1)}"
            
            journal = str(metadata.get('journal', ''))
            if journal and len(journal) > 30:
                journal = journal[:27] + "..."
            journal_info = f" | {journal}" if journal else ""
            
            metadata_info = f"{year}{journal_info}"
        
        print(f"{i:2d}. {title}")
        print(f"    ID: {doc_id}")
        print(f"    Score: {score:.3f} {stars}{metadata_info}")
        
        # Show text preview for top 3 results
        if i <= 3:
            text = doc_manager.get_document_text(doc_id)
            if text:
                # Get search terms for highlighting
                query_words = preprocess_query(query)
                search_terms = [term for sublist in query_words for term in sublist]
                term_patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in search_terms]
                
                # Split into lines
                lines = text.split('\n')
                # Find lines containing search terms
                matching_lines = []
                for idx, line in enumerate(lines):
                    if any(pattern.search(line) for pattern in term_patterns):
                        matching_lines.append(idx)
                
                if matching_lines:
                    # Get the first match and surrounding lines (3 before and 3 after)
                    match_idx = matching_lines[0]
                    start = max(0, match_idx - 3)
                    end = min(len(lines), match_idx + 4)  # +4 to include the match + 3 after
                    snippet_lines = lines[start:end]
                    
                    # Highlight terms in the snippet
                    highlighted_lines = []
                    for line in snippet_lines:
                        highlighted = line
                        for pattern in term_patterns:
                            highlighted = pattern.sub(lambda m: f'**{m.group(0)}**', highlighted)
                        highlighted_lines.append(highlighted)
                    
                    # Show up to 4 lines
                    for j, line in enumerate(highlighted_lines[:4]):
                        if len(line) > 120:
                            line = line[:117] + "..."
                        print(f"    {line}")
                else:
                    # Fallback to first lines if no matches
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    if lines:
                        preview = lines[0]
                        if len(preview) > 120:
                            preview = preview[:117] + "..."
                        print(f"    {preview}")
        
        print()
    
    # Summary
    print(f"{'─'*60}")
    print(f"📈 Search Statistics:")
    print(f"   • Documents found: {total_docs}")
    if results:
        print(f"   • Top score: {results[0][1]:.2f}")
        avg_score = sum(s for _, s in results) / len(results)
        print(f"   • Average score: {avg_score:.2f}")
    
    if total_docs > display_count:
        print(f"   • {total_docs - display_count} more documents not shown")
    
    print(f"{'='*80}")

# ============================================================================
# UPDATED INTERACTIVE SEARCH
# ============================================================================

def interactive_search():
    """Interactive search interface with ranking."""
    print("\n" + "="*80)
    print("🔬 CORD-19 SEARCH SYSTEM WITH ADVANCED RANKING")
    print("="*80)
    print("Features:")
    print("• Single & multi-word queries")
    print("• Citation-based ranking")
    print("• Journal prestige scoring")
    print("• Author prominence analysis")
    print("• Publication recency weighting")
    print("="*80)
    
    # Initialize components
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    
    # Initialize ranking system
    global ranker
    ranker = initialize_ranking_system()
    
    print(f"\n⚡ System initialized and ready")
    print(f"📊 {len(doc_manager.title_cache)} document titles loaded")
    
    if ranker:
        print(f"🧮 Advanced ranking system active")
    else:
        print(f"⚠️  Advanced ranking not available (using basic ranking)")
    
    last_results = []
    last_query = ""
    
    while True:
        try:
            user_input = input("\n🔎 Enter search query (or 'help' for commands): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n📖 Available commands:")
                print("  [query]           - Search with advanced ranking")
                print("  view [number]     - View detailed document")
                print("  simple [query]    - Search without advanced ranking")
                print("  weights           - Show ranking weights")
                print("  performance       - Run performance test")
                print("  clear             - Clear screen")
                print("  help              - Show this help")
                print("  quit              - Exit program")
                continue
            
            if user_input.lower() == 'clear':
                print("\n" * 50)
                continue
            
            if user_input.lower() == 'weights' and ranker:
                print("\n⚖️  Current Ranking Weights:")
                for factor, weight in ranker.weights.items():
                    print(f"  • {factor}: {weight:.1%}")
                continue
            
            if user_input.lower() == 'performance':
                performance_test()
                continue
            
            if user_input.lower().startswith('view '):
                if not last_results:
                    print("❌ No previous search results to view")
                    continue
                
                try:
                    result_num = int(user_input.split()[1])
                    if 1 <= result_num <= len(last_results):
                        doc_id, _ = last_results[result_num - 1]
                        show_detailed_document_view(doc_id, last_query)
                    else:
                        print(f"❌ Invalid result number. Use 1-{len(last_results)}")
                except (ValueError, IndexError):
                    print("❌ Usage: view [result_number]")
                continue
            
            if user_input.lower().startswith('simple '):
                # Simple search without advanced ranking
                simple_query = user_input[7:].strip()
                print(f"\n🔍 Simple search: '{simple_query}'")
                
                start_time = time.time()
                words = preprocess_query(simple_query)
                
                if not words:
                    print("⚠️  No valid search terms")
                    continue
                
                if len(words) == 1:
                    search_term = words[0][0]
                    doc_ids, frequencies = search_word(search_term)
                else:
                    results = multi_word_search(simple_query, 20)
                    doc_ids = [doc_id for doc_id, _ in results] if results else []
                    frequencies = None
                
                if doc_ids:
                    str_doc_ids = resolve_document_ids(doc_ids)
                    if frequencies and len(frequencies) == len(str_doc_ids):
                        results = [(doc_id, min(freq * 0.5, 10.0)) 
                                  for doc_id, freq in zip(str_doc_ids[:10], frequencies[:10])]
                        results.sort(key=lambda x: x[1], reverse=True)
                    else:
                        results = [(doc_id, 10.0 - (i * 0.1)) 
                                  for i, doc_id in enumerate(str_doc_ids[:10])]
                    
                    last_results = results
                    last_query = simple_query
                    
                    elapsed = time.time() - start_time
                    display_ranked_results(simple_query, results, max_display=10)
                    print(f"\n⏱️  Simple search completed in {elapsed:.3f}s")
                else:
                    print(f"\n📭 No documents found")
                continue
            
            if not user_input:
                continue
            
            # Perform advanced search with ranking
            last_query = user_input
            print(f"\n🔍 Searching with advanced ranking: '{user_input}'")
            
            start_time = time.time()
            last_results = smart_search_with_ranking(user_input, max_results=20)
            elapsed_time = time.time() - start_time
            
            if last_results:
                display_ranked_results(user_input, last_results, max_display=10)
                print(f"\n⏱️  Search completed in {elapsed_time:.3f} seconds")
                
                # Performance feedback
                word_count = len(preprocess_query(user_input))
                if word_count == 1:
                    if elapsed_time > 0.5:
                        print(f"⚠️  Single word query took {elapsed_time:.3f}s (target: <0.5s)")
                    else:
                        print(f"✅ Single word query under target: {elapsed_time:.3f}s")
                elif word_count == 5:
                    if elapsed_time > 1.5:
                        print(f"⚠️  5-word query took {elapsed_time:.3f}s (target: <1.5s)")
                    else:
                        print(f"✅ 5-word query under target: {elapsed_time:.3f}s")
            else:
                print(f"\n📭 No documents found in {elapsed_time:.3f}s")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)[:100]}")

# ============================================================================
# UPDATED PERFORMANCE TEST
# ============================================================================

def performance_test():
    """Performance test with ranking."""
    # Initialize
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    initialize_ranking_system()
    
    # Test cases
    test_cases = [
        ("covid", 1),
        ("coronavirus vaccine", 2),
        ("pandemic response strategies", 3),
        ("sars cov 2 transmission mask", 4),
        ("ventilator icu patients oxygen therapy", 5)
    ]
    
    print("\nRunning performance tests with ranking...")
    print(f"{'─'*80}")
    
    results = []
    all_passed = True
    
    for query, expected_words in test_cases:
        print(f"\nTesting: '{query}'")
        
        # Warm up
        _ = smart_search_with_ranking("covid", max_results=2)
        
        # Actual test
        start = time.time()
        search_results = smart_search_with_ranking(query, max_results=10)
        elapsed = time.time() - start
        
        found = len(search_results)
        
        # Check performance constraints
        status = "✅"
        if expected_words == 1:
            if elapsed > 0.5:
                status = "❌"
                all_passed = False
            elif elapsed > 0.4:
                status = "⚠️"
        elif expected_words == 5:
            if elapsed > 1.5:
                status = "❌"
                all_passed = False
            elif elapsed > 1.2:
                status = "⚠️"
        
        print(f"{status} {expected_words} word(s): {elapsed:.3f}s, found {found} docs")
        
        if search_results:
            doc_id, score = search_results[0]
            title = doc_manager.get_document_title(doc_id)
            if len(title) > 50:
                title = title[:47] + "..."
            print(f"   Top: {title} (score: {score:.2f})")
        print()
    
    print("="*80)
    if all_passed:
        print("✅ ALL PERFORMANCE TARGETS MET WITH RANKING!")
    else:
        print("⚠️  SOME TARGETS NOT MET")
    print("="*80)

# ============================================================================
# QUICK SEARCH FUNCTION (UPDATED)
# ============================================================================

def quick_search(query: str):
    """Quick search for command-line usage."""
    barrel_lookup.load_trie()
    doc_manager.load_metadata()
    initialize_ranking_system()
    
    print(f"\nSearching: '{query}'")
    start_time = time.time()
    
    results = smart_search_with_ranking(query, max_results=10)
    elapsed = time.time() - start_time
    
    if results:
        display_ranked_results(query, results, max_display=5)
        print(f"\nCompleted in {elapsed:.3f}s")
        
        word_count = len(preprocess_query(query))
        if word_count == 1 and elapsed > 0.5:
            print(f"Above 500ms target: {elapsed:.3f}s")
        elif word_count == 5 and elapsed > 1.5:
            print(f"Above 1.5s target: {elapsed:.3f}s")
    else:
        print(f"\nNo results found in {elapsed:.3f}s")

# ============================================================================
# DETAILED DOCUMENT VIEW (UPDATED)
# ============================================================================

def show_detailed_document_view(doc_id: str, query: str = ""):
    """Show detailed view of a document."""
    title = doc_manager.get_document_title(doc_id)
    
    print(f"\n{'='*80}")
    print(f"📄 DOCUMENT DETAILS")
    print(f"{'='*80}")
    print(f"Title: {title}")
    print(f"Document ID: {doc_id}")
    
    # Get ranking info if available
    global ranker
    if ranker and doc_id in ranker.paper_metadata:
        metadata = ranker.paper_metadata[doc_id]
        
        print(f"\n📊 Metadata:")
        if metadata.get('publish_time'):
            print(f"  • Published: {metadata['publish_time']}")
        if metadata.get('journal'):
            print(f"  • Journal: {metadata['journal']}")
        if metadata.get('authors'):
            authors = str(metadata['authors'])
            if len(authors) > 100:
                authors = authors[:97] + "..."
            print(f"  • Authors: {authors}")
        
        # Citation info
        if citation_builder:
            citations = citation_builder.get_citation_count(doc_id)
            print(f"  • Citations: {citations}")
    
    if query:
        print(f"\n🔍 Search query: '{query}'")
    
    # Text preview
    text = doc_manager.get_document_text(doc_id)
    if text:
        print(f"\n📝 Preview:")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i, line in enumerate(lines[:5], 1):
            if len(line) > 100:
                line = line[:97] + "..."
            print(f"{i}. {line}")
        
        if len(lines) > 5:
            print(f"... and {len(lines) - 5} more lines")
    else:
        print(f"\n📝 Text not available")
    
    print(f"\n{'='*80}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        quick_search(query)
        return
    
    print("\nSelect mode:")
    print("1. Interactive search with ranking")
    print("2. Performance testing")
    print("3. Simple search (no advanced ranking)")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        interactive_search()
    elif choice == "2":
        performance_test()
    elif choice == "3":
        query = input("Enter search query: ").strip()
        if query:
            # Simple search without ranking
            barrel_lookup.load_trie()
            doc_manager.load_metadata()
            
            print(f"\n🔍 Simple search: '{query}'")
            words = preprocess_query(query)
            
            if not words:
                print("⚠️  No valid search terms")
                return
            
            if len(words) == 1:
                doc_ids, _ = search_word(words[0][0])
            else:
                results = multi_word_search(query, 20)
                doc_ids = [doc_id for doc_id, _ in results] if results else []
            
            if doc_ids:
                str_doc_ids = resolve_document_ids(doc_ids[:10])
                results = [(doc_id, 10.0 - (i * 0.1)) for i, doc_id in enumerate(str_doc_ids)]
                display_ranked_results(query, results, max_display=10)
            else:
                print("📭 No documents found")
        else:
            print("❌ No query provided")
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()