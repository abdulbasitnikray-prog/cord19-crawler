"""
posting_list.py

Functions for creating posting lists (inverted index) from processed papers.

Posting lists store document-frequency pairs for each term,
enabling efficient Boolean and ranked retrieval.
"""

import json
import re
import os
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Any

# Text cleaning patterns
PUNCT_PATTERN = re.compile(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')))
SPACE_PATTERN = re.compile(r'\s+')
DIGIT_PATTERN = re.compile(r'\d+')

def clean_text(lines: List[str]) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        lines: List of text lines from a document
        
    Returns:
        Single cleaned string in lowercase
    """
    if not lines: 
        return ""
    
    full_text = " ".join(lines).lower()
    full_text = SPACE_PATTERN.sub(' ', full_text)
    full_text = PUNCT_PATTERN.sub('', full_text)
    full_text = DIGIT_PATTERN.sub('', full_text)
    
    return full_text.strip()

def extract_text(json_parse: Dict) -> List[str]:
    """
    Extract text content from CORD-19 paper JSON.
    """
    if json_parse is None:
        return []
    
    body = json_parse.get("body_text", [])
    lines = []
    
    for section in body:
        text = section.get("text", "")
        lines.extend(text.splitlines())
        if len(lines) >= 35:  
            break
    
    return lines[:35]

def process_paper_tokens(json_parse: Dict, nlp) -> Dict:
    """
    Process a single paper to extract lemmatized tokens.
    """
    if json_parse is None:
        return None
    
    raw_lines = extract_text(json_parse)
    if not raw_lines: 
        return None
    
    full_text = clean_text(raw_lines)
    if not full_text: 
        return None
    
    nlp.max_length = 1500000 
    doc = nlp(full_text) 
    indexed_tokens = []
    
    for token in doc:
        if (token.is_stop or token.is_punct or token.is_space or 
            token.like_num or len(token.text) < 2):
            continue
        
        token_data = {
            'lemma': token.lemma_
        }
        indexed_tokens.append(token_data)
    
    return {"tokens": indexed_tokens}

def create_posting_lists_for_doc(tokens_list: List[Dict], doc_id: str, 
                                 lexicon: Dict, word_id_counter: int) -> Tuple[Dict, List, List, Dict, int]:
    """
    Create posting lists for a single document.
    
    Args:
        tokens_list: List of token dictionaries
        doc_id: Document identifier
        lexicon: Current lexicon dictionary (word -> word_id)
        word_id_counter: Current highest word ID
        
    Returns:
        Tuple of (updated_lexicon, doc_word_ids, lemma_list, term_frequencies, new_word_id_counter)
    """
    doc_word_ids = []
    lemma_list = []
    term_frequencies = defaultdict(int)
    
    for token in tokens_list:
        lemma = token['lemma']
        lemma_list.append(lemma)
        
        # Update lexicon if new word
        if lemma not in lexicon:
            lexicon[lemma] = word_id_counter
            word_id_counter += 1
        
        # Get word ID
        word_id = lexicon[lemma]
        doc_word_ids.append(word_id)
        
        # Track term frequency
        term_frequencies[word_id] += 1
    
    return lexicon, doc_word_ids, lemma_list, term_frequencies, word_id_counter

def create_inverted_index(processed_papers: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create complete inverted index from processed papers.
    
    Args:
        processed_papers: List of processed paper dictionaries
        
    Returns:
        Tuple of (lexicon, forward_index, inverted_index, backward_index)
    """
    lexicon = {}
    forward_index = {}
    inverted_index = defaultdict(dict)
    backward_index = {}
    
    word_id_counter = 1
    
    for paper in processed_papers:
        doc_id = paper['cord_uid']
        tokens_list = paper['tokens']
        
        # Create posting lists for this document
        (lexicon, doc_word_ids, lemma_list, 
         term_frequencies, word_id_counter) = create_posting_lists_for_doc(
            tokens_list, doc_id, lexicon, word_id_counter
        )
        
        # Store in indexes
        forward_index[doc_id] = doc_word_ids
        backward_index[doc_id] = lemma_list
        
        # Update inverted index
        for word_id, freq in term_frequencies.items():
            inverted_index[word_id][doc_id] = freq
    
    return lexicon, forward_index, inverted_index, backward_index

def calculate_term_statistics(inverted_index: Dict, lexicon: Dict, total_docs: int) -> Dict:
    """
    Calculate statistics for each term.
    
    Args:
        inverted_index: Complete inverted index
        lexicon: Word to ID mapping
        total_docs: Total number of documents
        
    Returns:
        Dictionary with term statistics
    """
    term_stats = {}
    
    for word_id, postings in inverted_index.items():
        # Find word from lexicon
        word = next((w for w, wid in lexicon.items() if wid == word_id), None)
        
        if word:
            df = len(postings)  # Document Frequency
            cf = sum(postings.values())  # Collection Frequency
            idf = 0
            
            if df > 0:
                idf = total_docs / df
            
            term_stats[word_id] = {
                'word': word,
                'df': df,
                'cf': cf,
                'idf': idf,
                'postings': postings
            }
    
    return term_stats

def save_posting_lists(inverted_index: Dict, lexicon: Dict, 
                       output_dir: str = "posting_lists", max_files: int = 1000):
    """
    Save individual posting list files.
    
    Args:
        inverted_index: Complete inverted index
        lexicon: Word to ID mapping
        output_dir: Directory to save files
        max_files: Maximum number of files to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_docs = max([len(postings) for postings in inverted_index.values()] + [1])
    term_stats = calculate_term_statistics(inverted_index, lexicon, total_docs)
    saved_count = 0
    
    for word_id, stats in term_stats.items():
        if saved_count >= max_files:
            break
            
        word = stats['word']
        safe_word = "".join(c for c in word if c.isalnum() or c == '_')
        filename = os.path.join(output_dir, f"posting_{safe_word}_{word_id}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        saved_count += 1
    
    print(f"Saved {saved_count} posting list files to {output_dir}/")

def save_inverted_index(inverted_index: Dict, output_file: str = "inverted_index.json"):
    """
    Save complete inverted index to file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=None)
    print(f"Saved inverted index to {output_file} ({len(inverted_index):,} terms)")

if __name__ == "__main__":
    print("Posting List Module")
    print("=" * 50)
    print("\nFunctions for creating posting lists (inverted index).")