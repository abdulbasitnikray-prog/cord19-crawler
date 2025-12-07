import json 
import os 
import struct 
import pickle
from collections import defaultdict 
from . import barrel
import math

def varbyte_encode_num(n):
    """Encode a single number using VarByte encoding"""
    bytes_list = []
    while True:
        bytes_list.insert(0, n % 128)
        if n < 128:
            break
        n = n // 128
    bytes_list[-1] += 128  #continuation bit on last byte
    return bytes(bytes_list)

def varbyte_encode(numbers):
    """Encode a list of numbers using VarByte encoding"""
    encoded = bytearray()
    for n in numbers:
        encoded.extend(varbyte_encode_num(n))
    return encoded

def varbyte_decode(byte_arr):
    """Decode VarByte encoded data"""
    numbers = []
    curr = 0
    for byte in byte_arr:
        if byte < 128:
            curr = 128 * curr + byte
        else:
            curr = 128 * curr + (byte - 128)
            numbers.append(curr)
            curr = 0
    return numbers

def compress_posting_list(doc_ids, doc_freqs=None):
    """Compress a posting list with optional frequencies"""
    if not doc_ids:
        return {
            "type": "empty", 
            "data": b"",
            "doc_count": 0,
            "compressed_size": 0
        }
    
    #convert to integers and sort // doc_ids stored as strings
    if doc_freqs:
        #pair document IDs with frequencies
        paired = []
        for doc_id, freq in zip(doc_ids, doc_freqs):
            paired.append((int(doc_id), int(freq)))
        paired.sort(key=lambda x: x[0])  #sort by doc_id
        
        sorted_doc_ids = [p[0] for p in paired]
        sorted_doc_freqs = [p[1] for p in paired]
    else:
        sorted_doc_ids = sorted([int(d) for d in doc_ids])
        sorted_doc_freqs = None
    
    #calculate gaps
    if len(sorted_doc_ids) == 1:
        gaps = [sorted_doc_ids[0]]
    else:
        gaps = [sorted_doc_ids[0]]
        for i in range(1, len(sorted_doc_ids)):
            gaps.append(sorted_doc_ids[i] - sorted_doc_ids[i-1])
    
    #encode gaps
    encoded_gaps = varbyte_encode(gaps)
    
    result = {
        "doc_count": len(sorted_doc_ids),
        "compressed_size": len(encoded_gaps)
    }
    
    if doc_freqs and sorted_doc_freqs:
        #encode frequencies using delta encoding from mean
        avg_freq = sum(sorted_doc_freqs) / len(sorted_doc_freqs)
        freq_diffs = [int(f - avg_freq) for f in sorted_doc_freqs]
        encoded_freqs = varbyte_encode(freq_diffs)
        
        #combine everything
        combined_data = bytearray()
        combined_data.extend(struct.pack('I', len(encoded_gaps)))  # Gap length (4 bytes)
        combined_data.extend(encoded_gaps)                         # Encoded gaps
        combined_data.extend(struct.pack('d', avg_freq))           # Average frequency (8 bytes)
        combined_data.extend(struct.pack('I', len(encoded_freqs))) # Freq length (4 bytes)
        combined_data.extend(encoded_freqs)                        # Encoded frequency diffs
        
        result.update({
            "type": "with_freqs",
            "data": bytes(combined_data),
            "original_size": len(doc_ids) * 8,  
            "compression_ratio": (len(doc_ids) * 8) / len(combined_data) if combined_data else 1
        })
    else:
        result.update({
            "type": "no_freqs",
            "data": bytes(encoded_gaps),
            "original_size": len(doc_ids) * 4,  
            "compression_ratio": (len(doc_ids) * 4) / len(encoded_gaps) if encoded_gaps else 1
        })
    
    return result

def create_compressed_barrels(index_directory="indexes", 
                              barrel_dir="barrels", 
                              output_dir="compressed_barrels",
                              num_barrels=10):
    """Create barrels and then compress them WITH frequencies"""
    
    #first create barrels if they don't exist
    if not os.path.exists(barrel_dir) or not os.listdir(barrel_dir):
        print(f"Creating barrels from {index_directory}...")
        barrels, doc_to_barrels, word_to_barrel = barrel.create_and_save_barrels(
            index_directory=index_directory,
            output_dir=barrel_dir,
            num_barrels=num_barrels
        )
    
    #load inverted index for frequencies
    inverted_index_path = os.path.join(index_directory, "inverted_index.json")
    print(f"Loading inverted index for frequencies...")
    with open(inverted_index_path, 'r') as f:
        inverted_index = json.load(f)
    
    #create document ID mapping
    print("Creating document ID to integer mapping...")
    all_doc_ids = set()
    
    #collect all document IDs from inverted index
    for word_id, doc_freq_dict in inverted_index.items():
        all_doc_ids.update(doc_freq_dict.keys())
    
    # Create mapping (string -> integer)
    doc_id_mapping = {}
    int_to_str_mapping = {}
    
    for idx, doc_id in enumerate(sorted(all_doc_ids)):
        doc_id_mapping[doc_id] = idx + 1  # Start from 1
        int_to_str_mapping[idx + 1] = doc_id
    
    print(f"  Mapped {len(all_doc_ids):,} document IDs to integers")
    
    #save the mapping
    mapping_file = os.path.join(output_dir, "doc_id_mapping.pkl")
    with open(mapping_file, 'wb') as f:
        pickle.dump({
            "str_to_int": doc_id_mapping,
            "int_to_str": int_to_str_mapping
        }, f)
    print(f"  Saved ID mapping to {mapping_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load barrel mappings
    mappings_file = os.path.join(barrel_dir, "barrel_mappings.json")
    with open(mappings_file, 'r') as f:
        mappings = json.load(f)
    
    total_words = 0
    total_docs = 0
    
    # Process each barrel
    for barrel_file in os.listdir(barrel_dir):
        if barrel_file.startswith("barrel_") and barrel_file.endswith(".json"):
            barrel_path = os.path.join(barrel_dir, barrel_file)
            
            with open(barrel_path, 'r') as f:
                barrel_data = json.load(f)
            
            total_words += barrel_data["word_count"]
            
            # Compress posting lists in this barrel
            compressed_barrel = {
                "barrel_id": barrel_data["barrel_id"],
                "range_start": barrel_data["range_start"],
                "range_end": barrel_data["range_end"],
                "word_count": barrel_data["word_count"],
                "document_count": barrel_data["document_count"],
                "compressed_words": {}
            }
            
            words_processed = 0
            for word, word_data in barrel_data["words"].items():
                # Original string document IDs from barrel
                str_doc_ids = word_data["documents"]
                
                # Convert to integer IDs
                int_doc_ids = [doc_id_mapping[d] for d in str_doc_ids]
                
                total_docs += len(int_doc_ids)
                
                # Get word_id from barrel data
                word_id = word_data["word_id"]
                
                # Get frequencies from inverted index using word_id
                freq_dict = inverted_index.get(word_id, {})
                
                # Create parallel frequency list
                doc_freqs = []
                for str_doc_id in str_doc_ids:
                    # Use original string ID to get frequency
                    freq = freq_dict.get(str(str_doc_id), 1)
                    doc_freqs.append(freq)
                
                # Compress with frequencies and INTEGER IDs
                compressed = compress_posting_list(int_doc_ids, doc_freqs)
                
                # Store both original and integer IDs for reference
                compressed_barrel["compressed_words"][word] = {
                    "word_id": word_id,
                    "original_doc_ids": str_doc_ids,  # Keep original for reference
                    "compressed_data": compressed,
                    "doc_freq": word_data["doc_freq"],
                    "total_freq": word_data["total_freq"]
                }
                
                words_processed += 1
                if words_processed % 1000 == 0:
                    print(f"Processed {words_processed:,} words...")
            
            # Save compressed barrel
            compressed_path = os.path.join(output_dir, f"compressed_{barrel_file.replace('.json', '.pkl')}")
            with open(compressed_path, 'wb') as f:
                pickle.dump(compressed_barrel, f)
            
            print(f"Compressed barrel {barrel_data['barrel_id']} with {barrel_data['word_count']:,} words")
    
    print(f"\nTotal statistics:")
    print(f"  Total words processed: {total_words:,}")
    print(f"  Total document references: {total_docs:,}")
    print(f"  Average documents per word: {total_docs/total_words:.1f}")
    print(f"\nCompressed barrels with frequencies saved to: {os.path.abspath(output_dir)}")

def main():
    """Create barrels and compress them"""
    print("Creating and compressing barrels")
    
    # Create barrels and compress them
    create_compressed_barrels(
        index_directory="indexes",
        barrel_dir="barrels",
        output_dir="compressed_barrels",
        num_barrels=10
    )
    
    # Test compression
    print("\n" + "=" * 70)
    print("Testing compression functions...")
    
    doc_ids = [1, 5, 10, 15, 20, 25]
    doc_freqs = [3, 5, 2, 4, 1, 6]
    
    print(f"Original doc IDs: {doc_ids}")
    print(f"Original frequencies: {doc_freqs}")
    
    compressed = compress_posting_list(doc_ids, doc_freqs)
    
    print(f"\nCompression results:")
    print(f"  Type: {compressed['type']}")
    print(f"  Original size: {compressed['original_size']} bytes")
    print(f"  Compressed size: {compressed['compressed_size']} bytes")
    print(f"  Compression ratio: {compressed['compression_ratio']:.2f}x")
    print(f"  Document count: {compressed['doc_count']}")

if __name__ == "__main__":
    main()

