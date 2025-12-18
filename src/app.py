from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import os

# Import your existing modules
from multiwordSearch import multi_word_search, barrel_lookup, doc_manager
import singlewordSearch
from autocomplete import AutocompleteEngine
from semantic_search import semantic_engine
from dynamic_indexer import DynamicIndexer

app = Flask(__name__)
CORS(app)

def check_required_files():
    """Check if all required files exist before starting"""
    required_files = [
        os.path.join("data", "barrels", "barrel_mappings.json"),
        os.path.join("data", "processed_corpus.csv"),
        os.path.join("data", "indexes", "lexicon.json"),
        os.path.join("data", "indexes", "forward_index.json"),
        os.path.join("data", "indexes", "inverted_index.json"),
        os.path.join("data", "indexes", "backward_index.json")
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️ WARNING: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run these scripts first:")
        print("  1. python src/index.py")
        print("  2. python src/barreled_index.py")
        print("  3. python src/preprocess_papers.py")
        print("\nContinue anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            exit(1)
    
    return True

# --- Global Initialization ---
print("--- INITIALIZING SEARCH ENGINE ---")
check_required_files()
barrel_lookup.load_trie()
doc_manager.load_metadata()

autocomplete = AutocompleteEngine()
autocomplete.load_from_lexicon(os.path.join("data", "indexes", "lexicon.json"))

semantic_engine.load_model()
indexer = DynamicIndexer("data/indexes")
singlewordSearch.dynamic_indexer_ref = indexer

@app.route('/')
def home():
    total_docs = len(doc_manager.data) if doc_manager.data is not None else 0
    return render_template('index.html', total_docs=f"{total_docs:,}")

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    use_semantic = request.args.get('semantic', 'false').lower() == 'true'
    
    if not query:
        return jsonify({"error": "Empty query"}), 400

    start_time = time.time()
    
    # 1. Semantic Expansion
    query_words = query.split()
    synonyms_list = []
    final_search_query = query 

    if use_semantic:
        expansion_map = semantic_engine.expand_query(query_words)
        for word, syns in expansion_map.items():
            synonyms_list.extend(syns)
        synonyms_list = list(set(synonyms_list))
        if synonyms_list:
            final_search_query += " " + " ".join(synonyms_list)

    # 2. Perform Search
    results, total_found = multi_word_search(final_search_query, max_results=30)
    
    # 3. Format Results
    formatted_results = []
    for doc_id, score in results:
        title = doc_manager.get_document_title(doc_id)
        
        # --- CRITICAL FIX FOR PERFORMANCE ---
        # We DO NOT fetch text here anymore. It's too slow in Lite Mode.
        # We provide a generic snippet instead.
        snippet = "Click to view full document content..."
        
        formatted_results.append({
            "id": doc_id,
            "title": title,
            "score": f"{score:.2f}",
            "snippet": snippet,
            "match_terms": query_words
        })

    elapsed = time.time() - start_time
    
    return jsonify({
        "results": formatted_results,
        "total_hits": total_found,
        "time": f"{elapsed:.3f}",
        "synonyms": synonyms_list 
    })

@app.route('/api/autocomplete', methods=['GET'])
def get_autocomplete():
    prefix = request.args.get('q', '').strip()
    if not prefix or len(prefix) < 2:
        return jsonify([])
    suggestions = autocomplete.search(prefix)
    return jsonify(suggestions)

@app.route('/api/index', methods=['POST'])
def add_document():
    data = request.json
    if not data or 'content' not in data:
        return jsonify({"error": "Invalid data"}), 400
    
    if 'cord_uid' not in data:
        data['cord_uid'] = f"doc_{int(time.time())}"
        
    success = indexer.add_single_document(data)
    
    if success:
        doc_manager.add_dynamic_doc(
            data['cord_uid'], 
            data.get('title', 'Untitled'), 
            data['content']
        )
        return jsonify({"message": "Document indexed successfully", "id": data['cord_uid']})
    else:
        return jsonify({"error": "Failed to index document"}), 500

@app.route('/view/<doc_id>')
def view_document(doc_id):
    # Fetching text here is fine because it only happens ONCE per click
    title = doc_manager.get_document_title(doc_id)
    text = doc_manager.get_document_text(doc_id)
    
    if not text:
        text = "Error: Could not load the text for this document."
    
    return render_template('article.html', title=title, content=text, doc_id=doc_id)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)