"""
dynamic_indexer.py - Module for dynamic content addition to search index
Enables automatic indexing of newly added documents with proper ranking
"""

import os
import json as js
import time
from collections import defaultdict
from crawler import process_paper_single, clean_text, extract_text
from index import save_index_files

class DynamicIndexer:
    """Handles dynamic addition of new documents to existing search index"""
    
    def __init__(self, index_dir="data/indexes"):
        """
        Initialize dynamic indexer
        
        Args:
            index_dir: Directory containing existing index files
        """
        self.index_dir = index_dir
        self.lexicon = None
        self.forward_index = None
        self.inverted_index = None
        self.backward_index = None
        self.word_id_counter = 1
        self.loaded = False
        
    def load_existing_indexes(self):
        """Load existing indexes for updates"""
        print(f"Loading indexes from {self.index_dir}...")
        
        try:
            # Load lexicon
            lexicon_path = os.path.join(self.index_dir, "lexicon.json")
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon_data = js.load(f)
            
            # Convert to internal format
            self.lexicon = {}
            for word, info in lexicon_data.items():
                self.lexicon[word] = {
                    "id": info["id"],
                    "lemma": info.get("lemma", word),
                    "pos_counts": defaultdict(int, info.get("pos_counts", {})),
                    "total_count": info.get("total_count", 0)
                }
            
            # Load forward index
            forward_path = os.path.join(self.index_dir, "forward_index.json")
            with open(forward_path, 'r', encoding='utf-8') as f:
                self.forward_index = js.load(f)
            
            # Load inverted index (convert keys to int)
            inverted_path = os.path.join(self.index_dir, "inverted_index.json")
            with open(inverted_path, 'r', encoding='utf-8') as f:
                inverted_data = js.load(f)
                self.inverted_index = {int(k): v for k, v in inverted_data.items()}
            
            # Load backward index
            backward_path = os.path.join(self.index_dir, "backward_index.json")
            with open(backward_path, 'r', encoding='utf-8') as f:
                self.backward_index = js.load(f)
            
            # Calculate next word ID
            if self.lexicon:
                self.word_id_counter = max([info["id"] for info in self.lexicon.values()]) + 1
            
            self.loaded = True
            print(f"✓ Loaded {len(self.forward_index)} documents, {len(self.lexicon)} unique words")
            
        except Exception as e:
            print(f"✗ Error loading indexes: {e}")
            raise
    
    def add_single_document(self, document_data):
        """
        Add a single document to the index dynamically
        
        Args:
            document_data: dict with {
                'cord_uid': str,
                'title': str,
                'content': str,
                'metadata': dict (optional)
            }
        
        Returns:
            bool: True if successful
        """
        if not self.loaded:
            self.load_existing_indexes()
        
        doc_id = document_data["cord_uid"]
        
        # Check if document already exists
        if doc_id in self.forward_index:
            print(f"Document {doc_id} already exists in index")
            return False
        
        print(f"\nAdding new document: {doc_id}")
        print(f"Title: {document_data.get('title', 'Untitled')[:50]}...")
        
        # Process the document (using crawler's logic)
        json_parse = {
            "metadata": {"title": document_data.get("title", "")},
            "body_text": [{"text": document_data["content"]}]
        }
        
        # Add abstract if provided in metadata
        if "abstract" in document_data.get("metadata", {}):
            json_parse["abstract"] = [{"text": document_data["metadata"]["abstract"]}]
        
        # Process using existing crawler logic
        processed = process_paper_single(json_parse, cord_uid=doc_id)
        
        if not processed or "tokens" not in processed:
            print(f"✗ Failed to process document {doc_id}")
            return False
        
        tokens = processed["tokens"]
        
        # Add to backward index
        self.backward_index[doc_id] = tokens
        
        # Update lexicon and forward index
        doc_word_ids = []
        word_freq = {}
        
        for token_info in tokens:
            lemma = token_info['lemma']
            pos = token_info['pos']
            
            # Use lemma as word key
            word_key = lemma
            
            # Add to lexicon if new
            if word_key not in self.lexicon:
                self.lexicon[word_key] = {
                    "id": self.word_id_counter,
                    "pos_counts": defaultdict(int),
                    "lemma": lemma
                }
                self.word_id_counter += 1
            
            # Update POS counts
            self.lexicon[word_key]["pos_counts"][pos] += 1
            
            word_id = self.lexicon[word_key]["id"]
            doc_word_ids.append(word_id)
            word_freq[word_id] = word_freq.get(word_id, 0) + 1
        
        # Update forward index
        self.forward_index[doc_id] = doc_word_ids
        
        # Update inverted index
        for word_id, freq in word_freq.items():
            if word_id not in self.inverted_index:
                self.inverted_index[word_id] = {}
            self.inverted_index[word_id][doc_id] = freq
        
        print(f"✓ Successfully indexed document {doc_id}")
        print(f"  Words added: {len(doc_word_ids)}")
        print(f"  Unique words in doc: {len(set(doc_word_ids))}")
        
        return True
    
    def add_multiple_documents(self, documents_list):
        """
        Add multiple documents at once
        
        Args:
            documents_list: List of document_data dicts
        
        Returns:
            int: Number of successfully added documents
        """
        if not self.loaded:
            self.load_existing_indexes()
        
        print(f"\n{'='*60}")
        print(f"DYNAMIC INDEXING: Adding {len(documents_list)} new documents")
        print(f"{'='*60}")
        
        original_count = len(self.forward_index)
        success_count = 0
        
        for i, doc_data in enumerate(documents_list, 1):
            print(f"\n[{i}/{len(documents_list)}] Processing: {doc_data.get('cord_uid', 'Unknown')}")
            
            if self.add_single_document(doc_data):
                success_count += 1
        
        # Save updated indexes
        self.save_updated_indexes()
        
        new_count = len(self.forward_index)
        print(f"\n{'='*60}")
        print(f"DYNAMIC INDEXING COMPLETE")
        print(f"{'='*60}")
        print(f"Original documents: {original_count}")
        print(f"New documents added: {success_count}")
        print(f"Total documents now: {new_count}")
        print(f"Unique words: {len(self.lexicon)}")
        print(f"{'='*60}")
        
        return success_count
    
    def save_updated_indexes(self):
        """Save all updated indexes to disk"""
        if not self.loaded:
            print("No indexes loaded to save")
            return
        
        print(f"\nSaving updated indexes to {self.index_dir}...")
        
        # Convert lexicon for serialization
        lexicon_serializable = {}
        for word, info in self.lexicon.items():
            lexicon_serializable[word] = {
                "id": info["id"],
                "lemma": info.get("lemma", word),
                "pos_counts": dict(info["pos_counts"]),
                "total_count": sum(info["pos_counts"].values())
            }
        
        # Save using existing index.py function
        save_index_files(
            lexicon_serializable,
            self.forward_index,
            self.inverted_index,
            self.backward_index,
            self.index_dir
        )
        
        print(f"✓ Indexes saved successfully")
    
    def get_index_stats(self):
        """Get current index statistics"""
        if not self.loaded:
            return None
        
        return {
            "total_documents": len(self.forward_index),
            "unique_words": len(self.lexicon),
            "inverted_index_terms": len(self.inverted_index),
            "backward_index_docs": len(self.backward_index)
        }
    
    def search_dynamic_word(self, word):
        """
        Search for a word ONLY in the newly added dynamic documents.
        Returns: (doc_ids_list, freqs_list)
        """
        if not self.loaded:
            return [], []
        
        # Check if word exists in our dynamic lexicon
        if word not in self.lexicon:
            return [], []
            
        word_id = self.lexicon[word]["id"]
        
        # Check inverted index for this word ID
        if word_id in self.inverted_index:
            doc_freq_map = self.inverted_index[word_id] # {doc_id: freq}
            
            # Return keys (Doc IDs) and values (Frequencies)
            return list(doc_freq_map.keys()), list(doc_freq_map.values())
            
        return [], []


# ============ UTILITY FUNCTIONS ============

def add_document_from_csv(csv_path, index_dir="indexes"):
    """
    Utility function to add documents from a CSV file
    CSV should follow the same format as metadata.csv
    
    Args:
        csv_path: Path to CSV file
        index_dir: Index directory
    
    Returns:
        int: Number of documents added
    """
    import pandas as pd
    
    print(f"\nReading new documents from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        new_documents = []
        
        for _, row in df.iterrows():
            # Extract content
            content_parts = []
            
            if 'abstract' in row and pd.notna(row['abstract']):
                content_parts.append(str(row['abstract']))
            
            if 'body_text' in row and pd.notna(row['body_text']):
                content_parts.append(str(row['body_text']))
            
            if 'title' in row and pd.notna(row['title']):
                # Use title as content if nothing else
                if not content_parts:
                    content_parts.append(str(row['title']))
            
            content = " ".join(content_parts)
            
            if content.strip():
                doc_data = {
                    "cord_uid": row.get('cord_uid', f"NEW_{int(time.time())}_{_}"),
                    "title": row.get('title', 'New Document'),
                    "content": content,
                    "metadata": {
                        "source": row.get('source_x', 'user_upload'),
                        "authors": row.get('authors', ''),
                        "publish_time": row.get('publish_time', ''),
                        "journal": row.get('journal', ''),
                        "abstract": row.get('abstract', '')
                    }
                }
                new_documents.append(doc_data)
        
        if new_documents:
            indexer = DynamicIndexer(index_dir)
            added = indexer.add_multiple_documents(new_documents)
            return added
        else:
            print("✗ No valid content found in CSV file")
            return 0
            
    except Exception as e:
        print(f"Error processing CSV: {e}")
        import traceback
        traceback.print_exc()
        return 0


def add_single_document_quick(content, title="New Document", doc_id=None, index_dir="indexes"):
    """
    Quick function to add a single document
    
    Args:
        content: Document text content
        title: Document title
        doc_id: Document ID (auto-generated if None)
        index_dir: Index directory
    
    Returns:
        bool: True if successful
    """
    if doc_id is None:
        doc_id = f"USER_{int(time.time())}"
    
    document_data = {
        "cord_uid": doc_id,
        "title": title,
        "content": content,
        "metadata": {
            "source": "user_upload",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    indexer = DynamicIndexer(index_dir)
    return indexer.add_single_document(document_data)


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    # Example 1: Add a single document
    print("Example 1: Adding single document")
    
    new_doc = {
        "cord_uid": "TEST_DOC_001",
        "title": "COVID-19 Vaccine Efficacy Study 2024",
        "content": """
        This study examines the efficacy of new COVID-19 vaccines against emerging variants.
        We conducted clinical trials with 5000 participants over 6 months. Results show 
        that the new mRNA vaccine maintains 85% efficacy against Omicron variants.
        Adverse effects were minimal and comparable to previous versions.
        """,
        "metadata": {
            "authors": "John Smith, Sarah Johnson",
            "year": "2024",
            "journal": "Medical Research Journal"
        }
    }
    
    indexer = DynamicIndexer("data/indexes")
    if indexer.add_single_document(new_doc):
        print(" Document added successfully!")
        print(f"New stats: {indexer.get_index_stats()}")
    
    # Example 2: Add from CSV
    # add_document_from_csv("new_research_papers.csv")
    
    print("\nNew documents will now appear in search results with proper ranking!")