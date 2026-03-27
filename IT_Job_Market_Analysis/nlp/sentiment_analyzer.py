import pandas as pd
import os
import re
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_PATH = os.path.join(BASE_DIR, "data", "processed", "dynamic_lexicon.json")

NEGATIONS = {"không", "chẳng", "chưa", "đếch"}
INTENSIFIERS = {"rất", "quá", "cực", "siêu", "hơi", "hết"}
SUFFIX_INTENSIFIERS = {"quá", "lắm"}
STOPWORDS = {"và", "thì", "là", "mà", "có", "các", "những", "cho", "chiếc", "của", "để", "như", "này", "cũng", "được", "với", "app", "ứng_dụng", "nó", "mình", "người", "ra"}

def load_lexicon():
    pos_txt = os.path.join(BASE_DIR, "data", "lexicon", "vn_positive.txt")
    neg_txt = os.path.join(BASE_DIR, "data", "lexicon", "vn_negative.txt")

    pos_mem = set()
    neg_mem = set()
    
    # 1. Load Pre-trained Base Lexicons (from the large `.txt` banks)
    if os.path.exists(pos_txt):
        with open(pos_txt, "r", encoding="utf-8") as f:
            pos_mem.update({line.strip().lower() for line in f if line.strip()})
    if os.path.exists(neg_txt):
        with open(neg_txt, "r", encoding="utf-8") as f:
            neg_mem.update({line.strip().lower() for line in f if line.strip()})

    # 2. Add AI Self-Learned Lexicons (from `dynamic_lexicon.json`)
    if os.path.exists(LEXICON_PATH):
        try:
            with open(LEXICON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                pos_mem.update(set(data.get("positive", [])))
                neg_mem.update(set(data.get("negative", [])))
        except:
            pass
    else:
        # Guarantee JSON persistence initialization
        os.makedirs(os.path.dirname(LEXICON_PATH), exist_ok=True)
        save_lexicon(pos_mem, neg_mem)

    return pos_mem, neg_mem

def save_lexicon(pos_set, neg_set):
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        json.dump({"positive": list(pos_set), "negative": list(neg_set)}, f, ensure_ascii=False, indent=4)

def compound_phrases(text, phrase_list):
    for phrase in phrase_list:
        if " " in phrase:
            text = text.replace(phrase, phrase.replace(" ", "_"))
    return text

def preprocess_text(text, pos_words, neg_words):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
    text = re.sub(r'[.,!?;\n]', ' PUNCT ', text)
    
    all_phrases = list(pos_words) + list(neg_words) + ["không được", "ứng dụng"]
    all_phrases.sort(key=lambda x: len(x), reverse=True)
    text = compound_phrases(text, all_phrases)
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens), tokens

def get_sentiment(text):
    pos_set_raw, neg_set_raw = load_lexicon()
    cleaned_text, tokens = preprocess_text(text, pos_set_raw, neg_set_raw)
    
    pos_score = 0
    neg_score = 0
    final_features = []
    
    pos_set = {w.replace(" ", "_") for w in pos_set_raw}
    neg_set = {w.replace(" ", "_") for w in neg_set_raw}
    neg_set.add("không_được")
    
    for i, word in enumerate(tokens):
        if word == "PUNCT": continue
            
        if word in pos_set or word in neg_set:
            modifier_prefix = ""
            modifier_suffix = ""
            multiplier = 1
            flip = False
            
            if i > 0:
                prev = tokens[i-1]
                if prev == "PUNCT": pass
                elif prev in NEGATIONS:
                    flip = True
                    modifier_prefix = prev + " "
                elif prev in INTENSIFIERS:
                    multiplier = 2
                    modifier_prefix = prev + " "
                    
            if i < len(tokens) - 1:
                next_w = tokens[i+1]
                if next_w == "PUNCT": pass
                elif next_w in SUFFIX_INTENSIFIERS:
                    multiplier = 2
                    modifier_suffix = " " + next_w
                    
            display_word = word.replace("_", " ")
            phrase = f"{modifier_prefix}{display_word}{modifier_suffix}".strip()
            
            if word in pos_set:
                if flip:
                    neg_score += 1 * multiplier
                    final_features.append((phrase, "Negative"))
                else:
                    pos_score += 1 * multiplier
                    final_features.append((phrase, "Positive"))
                    
            elif word in neg_set:
                if flip:
                    pos_score += 0.5 * multiplier 
                    final_features.append((phrase, "Positive"))
                else:
                    neg_score += 1 * multiplier
                    final_features.append((phrase, "Negative"))
                    
    polarity = (pos_score - neg_score) / (pos_score + neg_score + 0.0001)
    if polarity > 0.05: sentiment = "Positive"
    elif polarity < -0.05: sentiment = "Negative"
    else: sentiment = "Neutral"
    return sentiment, polarity, final_features

def auto_train_lexicon(csv_path):
    """ Active Learning Agent that scans new data and expands vocabulary DB"""
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    if 'true_score' not in df.columns: return
    df = df.dropna(subset=['true_score', 'review_text'])
    
    # 4-5 stars are POS labels. 1-2 stars are NEG labels.
    pos_docs = df[df['true_score'] >= 4]['review_text'].tolist()
    neg_docs = df[df['true_score'] <= 2]['review_text'].tolist()
    
    pos_set_raw, neg_set_raw = load_lexicon()
    
    from collections import Counter
    def extract_unknown_words(docs):
        counter = Counter()
        for doc in docs:
            _, tokens = preprocess_text(doc, pos_set_raw, neg_set_raw)
            for t in tokens:
                if t != "PUNCT" and t not in pos_set_raw and t not in neg_set_raw and t not in NEGATIONS and t not in INTENSIFIERS and t not in SUFFIX_INTENSIFIERS:
                    counter[t] += 1
        return counter
        
    pos_counts = extract_unknown_words(pos_docs)
    neg_counts = extract_unknown_words(neg_docs)
    
    new_pos = []
    new_neg = []
    
    all_words = set(pos_counts.keys()).union(set(neg_counts.keys()))
    for w in all_words:
        p = pos_counts[w]
        n = neg_counts[w]
        total = p + n
        
        # Trigger condition: Seen at least 3 times in total, and 85% correlated towards one sentiment
        if total >= 3:
            if p / total >= 0.85:
                new_pos.append(w.replace("_", " "))
            elif n / total >= 0.85:
                new_neg.append(w.replace("_", " "))
                
    if new_pos or new_neg:
        pos_set_raw.update(new_pos)
        neg_set_raw.update(new_neg)
        save_lexicon(pos_set_raw, neg_set_raw)
        print(f"[CONTINUOUS LEARNING ML] Lexicon successfully updated! Added {len(new_pos)} Pos, {len(new_neg)} Neg words.")

def process_static_dataset(): ... 
