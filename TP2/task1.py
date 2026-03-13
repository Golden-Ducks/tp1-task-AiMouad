#  N-grams + BoW + KMeans

import numpy as np 
from sklean.cluster import KMeans
import re 
from itertools import product
from itertools import permutations


doc1 = "THe gold medal price is high rffort"
doc2 = "Winning a gold medal needs a hugh jump"
doc3 = "Market for a gold medal is a trade of sweat"
doc4 = "The athlete will trade all for a gold medal"

# class 2 : 

doc5 = "The gold bars price is high today"
doc6 = "Investing in gold bars needs a hight rate"
doc7 = "Market for gold bars is a trade of money"
doc8 = "The bank will trade all for gold bars"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    tokens = text.split()
    return tokens

def extract_unigrams(tokens):
    return [" ".join(tokens[i:i+1]) for i in range(len(tokens))]

def vectorize(docs,n_gram_size=1):
   
    # Step 1: preprocess all docs
    tokenized = [preprocess_text(doc) for doc in docs]

    # Step 2 : extract n-grams for each doc
    if n_gram_size == 1:
        doc_ngrams = [extract_unigrams(tokens) for tokens in tokenized]
    else:
        doc_ngrams = [extract_unigrams(tokens, n_gram_size) for tokens in tokenized]

    # Step 3 build soted vocab
    vocab = sorted(set(ng for ngrams in doc_ngrams for ng in ngrams))
    vocab_index = {ng: i for i, ng in enumerate(vocab)}

    print(f"\n[{n_gram_size}-gram] Vocabulary ({len(vocab)} terms):")
    for i, term in enumerate(vocab):
        print(f"  {i}: '{term}'")

    # Step 4 boolen vectorization
    x = np.zeros((len(docs), len(vocab)), dtype=int)
    for d,ng in enumerate(doc_ngrams):
        for n in ng:
            if n in vocab_index:
                x[d][vocab_index[n]] = 1  

    return x

# trainig and clustering

all_docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]
true_labels = [0, 0, 0, 0, 1, 1, 1, 1]

# 1 gram Experiment
x1 = vectorize(all_docs, n_gram_size=1)
km1 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x1)
# 2 gram Experiment
x2 = vectorize(all_docs, n_gram_size=2)
km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x2)


print(f"\n1 gram clusters: {km1.labels_}")
print(f"2 gram clusters: {km2.labels_}")


# ── Accuracy helper (handles label flip) ──────────────────────────

# Accuracy function : 

def cluster_accuracy(true, pred):
    best = 0 
    for p in permutations(set(pred)):
        mapping = {old: new for old, new in zip(sorted(set(pred)), p)}
        mapped = [mapping[p] for p in pred]
        acc = sum(t == m for t, m in zip(true, mapped)) / len(true)
        best = max(best, acc)
        return best

acc1 = cluster_accuracy(true_labels, km1.labels_)
acc2 = cluster_accuracy(true_labels, km2.labels_)

print(f"\n1 gram accuracy: {acc1:.2f}")
print(f"2 gram accuracy: {acc2:.2f}")