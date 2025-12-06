import json 
import os 
import math 
from collections import defaultdict, Counter
import time

def load_index_files(index_directory="output"):
    print(f"loading index files from {index_directory}")

    indexes={}
    files_to_load = {
        lexicon: "lexicon.json",
        forward_index: "forward_index.json",
        inverted_index: "inverted_index.json",
        backward_index: "backward_index.json",
    }

    for name, f_name in files_to_load.items():
        filepath = os.path.join(index_directory, f_name)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    indexes[name] = json.load(f)
                print(f"Loaded {name} from {filepath}")
            except Exception as e:
                print(f"Error loading {name} from {filepath}: {e}")

    return indexes

def create_barrels_by_freq(inverted_index, lexicon, num_barrels=10):
    """this function is supposed to create barrels on the basis of term frequency
    and the words appearing more often will be distributed evenly in barrels"""

    word_frequencies ={}

    for word_id, doc_freq in inverted_index.items():
        word = next((w for w, wid in lexicon.items() if wid == word_id), None)
        if word:
            doc_count = len(doc_freq)
            total_freq=sum(doc_freq.values())
            word_frequencies[word] = {
                "word_id": word_id,
                "doc_freq": doc_count,
                "total_freq": total_freq,
                "documents": list(doc_freq.keys())
            }
    
    sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1]['doc_freq'], reverse=True)

    barrels = []
    for i in range(num_barrels):
        barrels.append({
            "barrel_id": i+1,
            "words": {},
            "documents": set(),
            "total_doc_freq": 0,
            "total_word_freq": 0
        })
    
    #this distributes the first n words from our sorted words list
    top_num_words = min(5000, len(sorted_words))

    for i, (word, stats) in enumerate(sorted_words[:top_num_words]):
        barrel_idx = i % num_barrels
        barrels[barrel_idx]["words"][word] = stats
        barrels[barrel_idx]["word_count"] += 1
        barrels[barrel_idx]["total_doc_freq"] += stats["doc_frequency"]

    for word, stats in sorted_words[top_num_words:]:
        barrel_idx = min(range(num_barrels), key=lambda i: barrels[i]["total_doc_freq"])
        barrels[barrel_idx]["words"][word] = stats
        barrels[barrel_idx]["word_count"] += 1
        barrels[barrel_idx]["total_doc_freq"] += stats["doc_frequency"]

    
    #word to barrel map
    word_to_barrel = {}
    for barrel in barrels:
        for word in barrel["words"]:
            word_to_barrel[word] = barrel["barrel_id"]
    
    doc_to_barrel=defaultdict
    for word, stats in word_frequencies.items():
        barrel_id = word_to_barrel.get(word)
        if barrel_id:
            for doc_id in stats["documents"]:
                doc_to_barrel[doc_id].add(barrel_id)

                for barrel in barrels:
                    if barrel["barrel_id"] == barrel_id:  
                        barrel["documents"].add(doc_id)
    
    #convert document sets to lists
    for barrel in barrels:
        barrel["documents"] = sorted(list(barrel["documents"]))
        barrel["document_count"] = len(barrel["documents"])

    doc_to_barrels_serializable = {  #
        doc_id: sorted(list(barrel_ids))  
        for doc_id, barrel_ids in doc_to_barrel.items()
    }
    
    # range info to barrels
    for barrel in barrels:
        if barrel["words"]:  
            words_list = list(barrel["words"].keys())  
            barrel["range_start"] = min(words_list)  
            barrel["range_end"] = max(words_list)
        else:
            barrel["range_start"] = ""
            barrel["range_end"] = ""
    
    return barrels, doc_to_barrels_serializable




