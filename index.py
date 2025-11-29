
import json as js 
import os
import spacy 
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def get_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

def clean_text(text):
    cleaned = []
    for line in text:
        if not line or not line.strip():
            continue 
        line = line.lower()
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'[{}]'.format(re.escape('"#$%&*+/<=>@[\\]^_`{|}~')), '', line)
        line = re.sub(r'\d+', '', line)
        if line.strip():  
            cleaned.append(line)
    return cleaned

def remove_stop_words(para_lines):
    nlp = get_nlp_model()
    if nlp is None:
        return para_lines
        
    cleaned_lines = []
    for line in para_lines:
        if not line.strip():  
            continue
        doc = nlp(line)
        filtered_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        cleaned_lines.append(" ".join(filtered_words))
    return cleaned_lines

def sentence_segments(text_lines):
    nlp = get_nlp_model()
    if nlp is None:
        return text_lines
        
    sentences = []
    full_text = ' '.join(text_lines)
    doc = nlp(full_text)

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if len(sentence_text) > 7:
            sentences.append(sentence_text)
    return sentences

def pos_lemmatization(sentences):  
    if isinstance(sentences, list):
        text = ' '.join(sentences)
    else:
        text = sentences
        
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        if token.is_alpha and not token.is_stop and not token.is_punct:
            tokens.append({
                'lemma': token.lemma_.lower(),
                'pos': token.pos_,  
                'index_key': f"{token.lemma_.lower()}_{token.pos_}",
                'original': token.text
            })
    
    return {
        'tokens': tokens,
        'total_tokens': len(tokens),
        'content_words_count': len([t for t in tokens if t['pos'] in ['NOUN', 'VERB', 'ADJ', 'ADV']])
    }

def extract_text(json_parse):
    if json_parse is None:
        return []
    
    body = json_parse.get("body_text", [])
    lines = []
    
    for section in body:
        text = section.get("text", "")
        lines.extend(text.splitlines())
        if len(lines) >= 3:  
            break

    return lines[:3]

def process_papers(json_parse, cord_uid):  
    if json_parse is None:
        print("Warning: No JSON parse data available")
        return
    
    unprocessed_lines = extract_text(json_parse)
    cleaned_lines = clean_text(unprocessed_lines)
    removed_stopword_lines = remove_stop_words(cleaned_lines)
    sentences = sentence_segments(removed_stopword_lines)
    lemmatized = pos_lemmatization(sentences)

    print(f"Processed {lemmatized['total_tokens']} tokens for {cord_uid}")
    return lemmatized

def process_chunk(chunk):
    results = []
    for paper in chunk:
        try:
            if paper["json_parse"] is not None:
                processed_data = process_papers(paper['json_parse'], paper['cord_uid'])
                paper["processed"] = processed_data
            results.append(paper)
        except Exception as e:
            print(f"Error processing paper {paper.get('cord_uid', 'unknown')}: {e}")
            results.append(paper)  
    return results

def process_parallel(papers, chunk_size=100):
    print(f"Processing {len(papers)} papers in parallel...")
    
    chunks = [papers[i:i + chunk_size] for i in range(0, len(papers), chunk_size)]
    processed_papers = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, chunk_result in enumerate(executor.map(process_chunk, chunks)):
            processed_papers.extend(chunk_result)
            print(f"Completed chunk {i+1}/{len(chunks)}")
    
    return processed_papers

def build_lexicon(papers):
    lexicon = {}
    word_id = 1
    
    print(f"Building lexicon from {len(papers)} processed papers...")
    
    for i, paper in enumerate(papers):
        if i % 100 == 0:
            print(f"Processing paper {i+1}/{len(papers)} for lexicon...")
            
        if "processed" in paper and "tokens" in paper["processed"]:
            for token in paper["processed"]["tokens"]:
                word_key = token['index_key']  
                
                if word_key not in lexicon:
                    lexicon[word_key] = {
                        'word_id': word_id,
                        'word': token['lemma'],
                        'pos': token['pos'],
                        'doc_frequency': 1,
                        'total_frequency': 1
                    }
                    word_id += 1
                else:
                    lexicon[word_key]['total_frequency'] += 1
    
    print(f"Built lexicon with {len(lexicon)} unique terms")
    return lexicon

def create_inverted_index(papers):
    inverted_index = {}
    for paper in papers:
        doc_id = paper["cord_uid"]
        if "processed" in paper and "tokens" in paper["processed"]:
            tokens = paper["processed"]["tokens"]
            term_freq = {}
            for token_data in tokens:
                term = token_data["lemma"]
                term_freq[term] = term_freq.get(term, 0) + 1
            
            for term, freq in term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = {}
                inverted_index[term][doc_id] = freq
    return inverted_index

def save_lexicon(lexicon, filename="lexicon.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        js.dump(lexicon, f, indent=2, ensure_ascii=False)
    print(f"Lexicon saved to {filename}")

def save_index_file(inverted_index, filename="inverted_index.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as saved_file:
            js.dump(inverted_index, saved_file, indent=2, ensure_ascii=False)
        print(f"Inverted index saved to {filename}")
        print(f"Index size: {len(inverted_index)} terms")
    except Exception as e:
        print(f"Error saving index: {e}")

def main():
    input_file = "crawled_papers.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. crawler.py needs to be run first.")
        return
    
    print(f"crawled papers from {input_file}.")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = js.load(f)
    
    print(f"Loaded {len(papers)} papers for indexing")

    processed_papers = process_parallel(papers, chunk_size=100)
    lexicon = build_lexicon(processed_papers)
    save_lexicon(lexicon)
    

    inverted_index = create_inverted_index(processed_papers)
    save_index_file(inverted_index, "inverted_index.json")
    
    print("Indexing complete!")

if __name__ == "__main__":
    main()