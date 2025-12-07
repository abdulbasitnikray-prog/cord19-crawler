[1mdiff --git a/barreled_index.py b/barreled_index.py[m
[1mdeleted file mode 100644[m
[1mindex 582ebd7..0000000[m
[1m--- a/barreled_index.py[m
[1m+++ /dev/null[m
[36m@@ -1,273 +0,0 @@[m
[31m-import json [m
[31m-import os [m
[31m-import struct [m
[31m-import pickle[m
[31m-from collections import defaultdict [m
[31m-import barrel[m
[31m-import math[m
[31m-[m
[31m-def varbyte_encode_num(n):[m
[31m-    """Encode a single number using VarByte encoding"""[m
[31m-    bytes_list = [][m
[31m-    while True:[m
[31m-        bytes_list.insert(0, n % 128)[m
[31m-        if n < 128:[m
[31m-            break[m
[31m-        n = n // 128[m
[31m-    bytes_list[-1] += 128  #continuation bit on last byte[m
[31m-    return bytes(bytes_list)[m
[31m-[m
[31m-def varbyte_encode(numbers):[m
[31m-    """Encode a list of numbers using VarByte encoding"""[m
[31m-    encoded = bytearray()[m
[31m-    for n in numbers:[m
[31m-        encoded.extend(varbyte_encode_num(n))[m
[31m-    return encoded[m
[31m-[m
[31m-def varbyte_decode(byte_arr):[m
[31m-    """Decode VarByte encoded data"""[m
[31m-    numbers = [][m
[31m-    curr = 0[m
[31m-    for byte in byte_arr:[m
[31m-        if byte < 128:[m
[31m-            curr = 128 * curr + byte[m
[31m-        else:[m
[31m-            curr = 128 * curr + (byte - 128)[m
[31m-            numbers.append(curr)[m
[31m-            curr = 0[m
[31m-    return numbers[m
[31m-[m
[31m-def compress_posting_list(doc_ids, doc_freqs=None):[m
[31m-    """Compress a posting list with optional frequencies"""[m
[31m-    if not doc_ids:[m
[31m-        return {[m
[31m-            "type": "empty", [m
[31m-            "data": b"",[m
[31m-            "doc_count": 0,[m
[31m-            "compressed_size": 0[m
[31m-        }[m
[31m-    [m
[31m-    #convert to integers and sort // doc_ids stored as strings[m
[31m-    if doc_freqs:[m
[31m-        #pair document IDs with frequencies[m
[31m-        paired = [][m
[31m-        for doc_id, freq in zip(doc_ids, doc_freqs):[m
[31m-            paired.append((int(doc_id), int(freq)))[m
[31m-        paired.sort(key=lambda x: x[0])  #sort by doc_id[m
[31m-        [m
[31m-        sorted_doc_ids = [p[0] for p in paired][m
[31m-        sorted_doc_freqs = [p[1] for p in paired][m
[31m-    else:[m
[31m-        sorted_doc_ids = sorted([int(d) for d in doc_ids])[m
[31m-        sorted_doc_freqs = None[m
[31m-    [m
[31m-    #calculate gaps[m
[31m-    if len(sorted_doc_ids) == 1:[m
[31m-        gaps = [sorted_doc_ids[0]][m
[31m-    else:[m
[31m-        gaps = [sorted_doc_ids[0]][m
[31m-        for i in range(1, len(sorted_doc_ids)):[m
[31m-            gaps.append(sorted_doc_ids[i] - sorted_doc_ids[i-1])[m
[31m-    [m
[31m-    #encode gaps[m
[31m-    encoded_gaps = varbyte_encode(gaps)[m
[31m-    [m
[31m-    result = {[m
[31m-        "doc_count": len(sorted_doc_ids),[m
[31m-        "compressed_size": len(encoded_gaps)[m
[31m-    }[m
[31m-    [m
[31m-    if doc_freqs and sorted_doc_freqs:[m
[31m-        #encode frequencies using delta encoding from mean[m
[31m-        avg_freq = sum(sorted_doc_freqs) / len(sorted_doc_freqs)[m
[31m-        freq_diffs = [int(f - avg_freq) for f in sorted_doc_freqs][m
[31m-        encoded_freqs = varbyte_encode(freq_diffs)[m
[31m-        [m
[31m-        #combine everything[m
[31m-        combined_data = bytearray()[m
[31m-        combined_data.extend(struct.pack('I', len(encoded_gaps)))  # Gap length (4 bytes)[m
[31m-        combined_data.extend(encoded_gaps)                         # Encoded gaps[m
[31m-        combined_data.extend(struct.pack('d', avg_freq))           # Average frequency (8 bytes)[m
[31m-        combined_data.extend(struct.pack('I', len(encoded_freqs))) # Freq length (4 bytes)[m
[31m-        combined_data.extend(encoded_freqs)                        # Encoded frequency diffs[m
[31m-        [m
[31m-        result.update({[m
[31m-            "type": "with_freqs",[m
[31m-            "data": bytes(combined_data),[m
[31m-            "original_size": len(doc_ids) * 8,  [m
[31m-            "compression_ratio": (len(doc_ids) * 8) / len(combined_data) if combined_data else 1[m
[31m-        })[m
[31m-    else:[m
[31m-        result.update({[m
[31m-            "type": "no_freqs",[m
[31m-            "data": bytes(encoded_gaps),[m
[31m-            "original_size": len(doc_ids) * 4,  [m
[31m-            "compression_ratio": (len(doc_ids) * 4) / len(encoded_gaps) if encoded_gaps else 1[m
[31m-        })[m
[31m-    [m
[31m-    return result[m
[31m-[m
[31m-def create_compressed_barrels(index_directory="indexes", [m
[31m-                              barrel_dir="barrels", [m
[31m-                              output_dir="compressed_barrels",[m
[31m-                              num_barrels=10):[m
[31m-    """Create barrels and then compress them WITH frequencies"""[m
[31m-    [m
[31m-    #first create barrels if they don't exist[m
[31m-    if not os.path.exists(barrel_dir) or not os.listdir(barrel_dir):[m
[31m-        print(f"Creating barrels from {index_directory}...")[m
[31m-        barrels, doc_to_barrels, word_to_barrel = barrel.create_and_save_barrels([m
[31m-            index_directory=index_directory,[m
[31m-            output_dir=barrel_dir,[m
[31m-            num_barrels=num_barrels[m
[31m-        )[m
[31m-    [m
[31m-    #load inverted index for frequencies[m
[31m-    inverted_index_path = os.path.join(index_directory, "inverted_index.json")[m
[31m-    print(f"Loading inverted index for frequencies...")[m
[31m-    with open(inverted_index_path, 'r') as f:[m
[31m-        inverted_index = json.load(f)[m
[31m-    [m
[31m-    #create document ID mapping[m
[31m-    print("Creating document ID to integer mapping...")[m
[31m-    all_doc_ids = set()[m
[31m-    [m
[31m-    #collect all document IDs from inverted index[m
[31m-    for word_id, doc_freq_dict in inverted_index.items():[m
[31m-        all_doc_ids.update(doc_freq_dict.keys())[m
[31m-    [m
[31m-    # Create mapping (string -> integer)[m
[31m-    doc_id_mapping = {}[m
[31m-    int_to_str_mapping = {}[m
[31m-    [m
[31m-    for idx, doc_id in enumerate(sorted(all_doc_ids)):[m
[31m-        doc_id_mapping[doc_id] = idx + 1  # Start from 1[m
[31m-        int_to_str_mapping[idx + 1] = doc_id[m
[31m-    [m
[31m-    print(f"  Mapped {len(all_doc_ids):,} document IDs to integers")[m
[31m-    [m
[31m-    #save the mapping[m
[31m-    mapping_file = os.path.join(output_dir, "doc_id_mapping.pkl")[m
[31m-    with open(mapping_file, 'wb') as f:[m
[31m-        pickle.dump({[m
[31m-            "str_to_int": doc_id_mapping,[m
[31m-            "int_to_str": int_to_str_mapping[m
[31m-        }, f)[m
[31m-    print(f"  Saved ID mapping to {mapping_file}")[m
[31m-    [m
[31m-    os.makedirs(output_dir, exist_ok=True)[m
[31m-    [m
[31m-    # Load barrel mappings[m
[31m-    mappings_file = os.path.join(barrel_dir, "barrel_mappings.json")[m
[31m-    with open(mappings_file, 'r') as f:[m
[31m-        mappings = json.load(f)[m
[31m-    [m
[31m-    total_words = 0[m
[31m-    total_docs = 0[m
[31m-    [m
[31m-    # Process each barrel[m
[31m-    for barrel_file in os.listdir(barrel_dir):[m
[31m-        if barrel_file.startswith("barrel_") and barrel_file.endswith(".json"):[m
[31m-            barrel_path = os.path.join(barrel_dir, barrel_file)[m
[31m-            [m
[31m-            with open(barrel_path, 'r') as f:[m
[31m-                barrel_data = json.load(f)[m
[31m-            [m
[31m-            total_words += barrel_data["word_count"][m
[31m-            [m
[31m-            # Compress posting lists in this barrel[m
[31m-            compressed_barrel = {[m
[31m-                "barrel_id": barrel_data["barrel_id"],[m
[31m-                "range_start": barrel_data["range_start"],[m
[31m-                "range_end": barrel_data["range_end"],[m
[31m-                "word_count": barrel_data["word_count"],[m
[31m-                "document_count": barrel_data["document_count"],[m
[31m-                "compressed_words": {}[m
[31m-            }[m
[31m-            [m
[31m-            words_processed = 0[m
[31m-            for word, word_data in barrel_data["words"].items():[m
[31m-                # Original string document IDs from barrel[m
[31m-                str_doc_ids = word_data["documents"][m
[31m-                [m
[31m-                # Convert to integer IDs[m
[31m-                int_doc_ids = [doc_id_mapping[d] for d in str_doc_ids][m
[31m-                [m
[31m-                total_docs += len(int_doc_ids)[m
[31m-                [m
[31m-                # Get word_id from barrel data[m
[31m-                word_id = word_data["word_id"][m
[31m-                [m
[31m-                # Get frequencies from inverted index using word_id[m
[31m-                freq_dict = inverted_index.get(word_id, {})[m
[31m-                [m
[31m-                # Create parallel frequency list[m
[31m-                doc_freqs = [][m
[31m-                for str_doc_id in str_doc_ids:[m
[31m-                    # Use original string ID to get frequency[m
[31m-                    freq = freq_dict.get(str(str_doc_id), 1)[m
[31m-                    doc_freqs.append(freq)[m
[31m-                [m
[31m-                # Compress with frequencies and INTEGER IDs[m
[31m-                compressed = compress_posting_list(int_doc_ids, doc_freqs)[m
[31m-                [m
[31m-                # Store both original and integer IDs for reference[m
[31m-                compressed_barrel["compressed_words"][word] = {[m
[31m-                    "word_id": word_id,[m
[31m-                    "original_doc_ids": str_doc_ids,  # Keep original for reference[m
[31m-                    "compressed_data": compressed,[m
[31m-                    "doc_freq": word_data["doc_freq"],[m
[31m-                    "total_freq": word_data["total_freq"][m
[31m-                }[m
[31m-                [m
[31m-                words_processed += 1[m
[31m-                if words_processed % 1000 == 0:[m
[31m-                    print(f"Processed {words_processed:,} words...")[m
[31m-            [m
[31m-            # Save compressed barrel[m
[31m-            compressed_path = os.path.join(output_dir, f"compressed_{barrel_file.replace('.json', '.pkl')}")[m
[31m-            with open(compressed_path, 'wb') as f:[m
[31m-                pickle.dump(compressed_barrel, f)[m
[31m-            [m
[31m-            print(f"Compressed barrel {barrel_data['barrel_id']} with {barrel_data['word_count']:,} words")[m
[31m-    [m
[31m-    print(f"\nTotal statistics:")[m
[31m-    print(f"  Total words processed: {total_words:,}")[m
[31m-    print(f"  Total document references: {total_docs:,}")[m
[31m-    print(f"  Average documents per word: {total_docs/total_words:.1f}")[m
[31m-    print(f"\nCompressed barrels with frequencies saved to: {os.path.abspath(output_dir)}")[m
[31m-[m
[31m-def main():[m
[31m-    """Create barrels and compress them"""[m
[31m-    print("Creating and compressing barrels")[m
[31m-    [m
[31m-    # Create barrels and compress them[m
[31m-    create_compressed_barrels([m
[31m-        index_directory="indexes",[m
[31m-        barrel_dir="barrels",[m
[31m-        output_dir="compressed_barrels",[m
[31m-        num_barrels=10[m
[31m-    )[m
[31m-    [m
[31m-    # Test compression[m
[31m-    print("\n" + "=" * 70)[m
[31m-    print("Testing compression functions...")[m
[31m-    [m
[31m-    doc_ids = [1, 5, 10, 15, 20, 25][m
[31m-    doc_freqs = [3, 5, 2, 4, 1, 6][m
[31m-    [m
[31m-    print(f"Original doc IDs: {doc_ids}")[m
[31m-    print(f"Original frequencies: {doc_freqs}")[m
[31m-    [m
[31m-    compressed = compress_posting_list(doc_ids, doc_freqs)[m
[31m-    [m
[31m-    print(f"\nCompression results:")[m
[31m-    print(f"  Type: {compressed['type']}")[m
[31m-    print(f"  Original size: {compressed['original_size']} bytes")[m
[31m-    print(f"  Compressed size: {compressed['compressed_size']} bytes")[m
[31m-    print(f"  Compression ratio: {compressed['compression_ratio']:.2f}x")[m
[31m-    print(f"  Document count: {compressed['doc_count']}")[m
[31m-[m
[31m-if __name__ == "__main__":[m
[31m-    main()[m
[31m-[m
