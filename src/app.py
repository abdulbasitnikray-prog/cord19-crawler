from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
import os
import json

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
    # Adjust paths for Docker: go up one level from src/
    base_path = os.path.join(os.path.dirname(__file__), '..')
    
    required_files = [
        os.path.join(base_path, "data", "barrels", "barrel_mappings.json"),
        os.path.join(base_path, "data", "processed_corpus.csv"),
        os.path.join(base_path, "data", "indexes", "lexicon.json"),
        os.path.join(base_path, "data", "indexes", "forward_index.json"),
        os.path.join(base_path, "data", "indexes", "inverted_index.json"),
        os.path.join(base_path, "data", "indexes", "backward_index.json")
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
        
        # In production/Docker, don't prompt for input - just exit
        import sys
        if not sys.stdin.isatty():
            print("\n⚠️ Running in non-interactive mode. Exiting...")
            exit(1)
        
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

# Adjust path for Docker/production
base_path = os.path.join(os.path.dirname(__file__), '..')
lexicon_path = os.path.join(base_path, "data", "indexes", "lexicon.json")

autocomplete = AutocompleteEngine()
autocomplete.load_from_lexicon(lexicon_path)

semantic_engine.load_model()
indexer = DynamicIndexer(os.path.join(base_path, "data", "indexes"))
singlewordSearch.dynamic_indexer_ref = indexer

@app.route('/')
def home():
    # Show Total (Static + Dynamic)
    # Uses the total count from the CSV scan + any new dynamic docs
    static_count = getattr(doc_manager, 'total_docs_in_corpus', len(doc_manager.title_cache))
    # Count documents in the dynamic indexer if it exists
    dynamic_count = len(indexer.forward_index) if indexer else 0
    total = static_count + dynamic_count
    return render_template('index.html', total_docs=f"{total:,}")
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

# --- NEW: UNIFIED UPLOAD API (Manual + Files) ---
@app.route('/api/upload', methods=['POST'])
def upload_document():
    content = ""
    title = "Untitled"
    
    # 1. Handle File Uploads (JSON or TXT)
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        title = file.filename
        
        try:
            if file.filename.lower().endswith('.json'):
                data = json.load(file)
                # Support CORD-19 format (metadata + body_text)
                if 'metadata' in data and 'title' in data['metadata']:
                    title = data['metadata']['title']
                    content = " ".join([p['text'] for p in data.get('body_text', [])])
                # Support Simple format (title + content)
                elif 'content' in data:
                    title = data.get('title', title)
                    content = data['content']
                else:
                    return jsonify({"error": "Unknown JSON structure. Use CORD-19 or simple {'title':, 'content':} format"}), 400
            
            elif file.filename.lower().endswith('.txt'):
                content = file.read().decode('utf-8')
            
            else:
                return jsonify({"error": "Unsupported file type. Use .json or .txt"}), 400

        except Exception as e:
            return jsonify({"error": f"Failed to parse file: {str(e)}"}), 500

    # 2. Handle Manual Entry (JSON Body)
    elif request.json:
        data = request.json
        title = data.get('title', 'Untitled')
        content = data.get('content', '')
    else:
        return jsonify({"error": "No data provided"}), 400

    if not content or not content.strip():
        return jsonify({"error": "Document content is empty"}), 400

    # 3. Index It (Instant Delta Indexing)
    doc_id = f"dyn_{int(time.time())}"
    doc_data = {
        "cord_uid": doc_id,
        "title": title,
        "content": content
    }
    
    # Add to Dynamic Indexer (Search)
    if indexer.add_single_document(doc_data):
        # Add to Document Manager (View/Retrieval)
        doc_manager.add_dynamic_doc(doc_id, title, content)
        return jsonify({"message": "Indexed successfully", "id": doc_id, "title": title})
    else:
        return jsonify({"error": "Indexing failed (Duplicate or Empty)"}), 500
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