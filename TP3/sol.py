
import math

corpus = [
    "the cat sat on the mat","the dog sat on the log",
    "the cat and the dog are friends",
]

# STEP 1 — Tokenization

def tokenize(doc):

    return doc.lower().split()

# STEP 2 — Vocabulary
def build_vocabulary(corpus):
    
    vocab = set()
    for doc in corpus:
        for word in tokenize(doc):
            vocab.add(word)
    return sorted(list(vocab))

# STEP 3 — TF (Term Frequency)

def compute_tf(doc):
    
    tokens = tokenize(doc)
    total = len(tokens)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    tf = {}
    for word, count in counts.items():
        tf[word] = count / total
    return tf


# STEP 4 — DF (Document Frequency)

def compute_df(corpus):
    
    df = {}
    for doc in corpus:
        unique_words = set(tokenize(doc))   # set → 1 count per doc
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    return df


# STEP 5 — IDF (Inverse Document Frequency + Smoothing)

def compute_idf(corpus):
    
    N = len(corpus)
    df = compute_df(corpus)
    idf = {}
    for word, freq in df.items():
        idf[word] = math.log((1 + N) / (1 + freq)) + 1
    return idf


# STEP 6 — TF-IDF

def compute_tfidf(corpus):
  
    idf = compute_idf(corpus)
    all_tfidf = []
    for doc in corpus:
        tf = compute_tf(doc)
        tfidf = {}
        for word, tf_score in tf.items():
            tfidf[word] = tf_score * idf[word]
        all_tfidf.append(tfidf)
    return all_tfidf


def to_vector(tfidf, vocab):
   
    return [tfidf.get(word, 0.0) for word in vocab]


# STEP 7 — Cosine Similarity

def cosine_similarity(vec_a, vec_b):
    """
    cos(A, B) = (A · B) / (||A|| × ||B||)

    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# MAIN — Run everything & print results

if __name__ == "__main__":

    SEP = "=" * 55

    # ── Vocabulary 
    print(SEP)
    print("— Vocabulary")
    print(SEP)
    vocab = build_vocabulary(corpus)
    print(f"Vocab ({len(vocab)} words): {vocab}\n")

    # ── TF 
    print(SEP)
    print("— Term Frequency (TF)")
    print(SEP)
    for i, doc in enumerate(corpus):
        tf = compute_tf(doc)
        print(f"Doc{i+1}: \"{doc}\"")
        for word, score in sorted(tf.items()):
            print(f"   TF('{word}') = {score:.4f}")
        print()

    # ── DF 
    print(SEP)
    print("— Document Frequency (DF)")
    print(SEP)
    df = compute_df(corpus)
    for word, freq in sorted(df.items(), key=lambda x: -x[1]):
        print(f"   DF('{word}') = {freq}")
    print()

    # ── IDF 
    print(SEP)
    print("— Inverse Document Frequency (IDF + Smoothing)")
    print(SEP)
    N = len(corpus)
    idf = compute_idf(corpus)
    for word, score in sorted(idf.items(), key=lambda x: x[1]):
        raw = math.log((1 + N) / (1 + df[word]))
        print(f"   IDF('{word}') = log({1+N}/{1+df[word]}) + 1"
              f" = {raw:.4f} + 1 = {score:.4f}")
    print()

    # ── TF-IDF 
    print(SEP)
    print("— TF-IDF Scores")
    print(SEP)
    all_tfidf = compute_tfidf(corpus)
    for i, tfidf in enumerate(all_tfidf):
        print(f"Doc{i+1}:")
        for word, score in sorted(tfidf.items(), key=lambda x: -x[1]):
            print(f"   TF-IDF('{word}') = {score:.4f}")
        print()

    # ── Vectors 
    print(SEP)
    print("— TF-IDF Vectors (aligned with vocab)")
    print(SEP)
    vectors = [to_vector(t, vocab) for t in all_tfidf]
    header = f"{'word':<10}" + "".join(f"  Doc{i+1}  " for i in range(len(corpus)))
    print(header)
    print("-" * len(header))
    for j, word in enumerate(vocab):
        row = f"{word:<10}"
        for vec in vectors:
            row += f"  {vec[j]:.4f}"
        print(row)
    print()

    # ── Cosine Similarity 
    print(SEP)
    print("— Cosine Similarity")
    print(SEP)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"   cos(Doc{i+1}, Doc{j+1}) = {sim:.4f}")
    print()

    # ── Full matrix 
    print("Full similarity matrix:")
    header = f"{'':>6}" + "".join(f"  Doc{i+1} " for i in range(len(corpus)))
    print(header)
    for i, va in enumerate(vectors):
        row = f"Doc{i+1}  "
        for j, vb in enumerate(vectors):
            row += f"  {cosine_similarity(va, vb):.4f}"
        print(row)
