import math
import string
import random

STOP_WORDS = {"the", "is", "a", "an", "in", "of", "to", "and"}

raw_corpus = [
    "Luna eats food.",
    "The dog loves food.",
    "The AI is massive.",
]

def clean(doc):
    doc = doc.lower()
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    return [w for w in doc.split() if w not in STOP_WORDS]

def build_vocab(corpus):
    vocab = set()
    for tokens in corpus:
        vocab.update(tokens)
    return sorted(list(vocab))

def compute_df(corpus):
    df = {}
    for tokens in corpus:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    return df

def compute_idf(df, N):
    return {word: math.log((N + 1) / (freq + 1)) + 1 for word, freq in df.items()}

def compute_tf(tokens):
    total = len(tokens)
    tf = {}
    for word in tokens:
        tf[word] = tf.get(word, 0) + 1
    return {word: count / total for word, count in tf.items()}

def to_vector(tf, idf, vocab):
    return [tf.get(w, 0.0) * idf.get(w, 0.0) for w in vocab]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


cleaned = [clean(doc) for doc in raw_corpus]
vocab = build_vocab(cleaned)
df = compute_df(cleaned)
N = len(cleaned)
idf = compute_idf(df, N)
all_tf = [compute_tf(tokens) for tokens in cleaned]
tfidf_vectors = [to_vector(tf, idf, vocab) for tf in all_tf]

print("=" * 55)
print("STEP 1 — Cleaned Corpus")
print("=" * 55)
for i, tokens in enumerate(cleaned):
    print(f"  Doc{i+1}: {tokens}")

print("\n" + "=" * 55)
print("STEP 2 — Vocabulary")
print("=" * 55)
print(f"  Vocab ({len(vocab)} words): {vocab}")

print("\n" + "=" * 55)
print("STEP 3 — Document Frequency (DF)")
print("=" * 55)
print(f"  {'Word':<12} {'Appears in':<25} DF")
print(f"  {'-'*12} {'-'*25} --")
for word in vocab:
    appears = [f"d{i+1}" for i, t in enumerate(cleaned) if word in t]
    print(f"  {word:<12} {', '.join(appears):<25} {df[word]}")

print("\n" + "=" * 55)
print("STEP 4 — IDF (Smoothed)")
print("=" * 55)
print(f"  Formula: log((N+1) / (DF+1)) + 1    N={N}\n")
print(f"  {'Word':<12} {'DF':<5} {'Calculation':<25} IDF")
print(f"  {'-'*12} {'-'*5} {'-'*25} -----")
for word in vocab:
    d = df[word]
    print(f"  {word:<12} {d:<5} log({N+1}/{d+1}) + 1{'':>12} {idf[word]:.4f}")

print("\n" + "=" * 55)
print("STEP 5 — TF (Term Frequency)")
print("=" * 55)
for i, (tokens, tf) in enumerate(zip(cleaned, all_tf)):
    total = len(tokens)
    print(f"\n  Doc{i+1}: {tokens}  (total={total})")
    print(f"  {'Word':<12} Count    TF")
    print(f"  {'-'*12} -----    ------")
    for word in tokens:
        count = int(round(tf[word] * total))
        print(f"  {word:<12} {count}/{total}      {tf[word]:.4f}")

print("\n" + "=" * 55)
print("STEP 6 — TF-IDF Vectors")
print("=" * 55)
print(f"  {'Word':<12}", end="")
for i in range(N):
    print(f"  {'Doc'+str(i+1):<10}", end="")
print()
print(f"  {'-'*12}", end="")
for i in range(N):
    print(f"  {'-'*10}", end="")
print()
for j, word in enumerate(vocab):
    print(f"  {word:<12}", end="")
    for i in range(N):
        val = tfidf_vectors[i][j]
        print(f"  {val:.4f}    ", end="")
    print()

query_raw = "AI food"
query_tokens = clean(query_raw)
query_tf = compute_tf(query_tokens)
query_vector = to_vector(query_tf, idf, vocab)

print("\n" + "=" * 55)
print("STEP 7 — Query + Cosine Similarity")
print("=" * 55)
print(f"  Query: \"{query_raw}\" -> {query_tokens}\n")
print(f"  Query TF-IDF vector:")
for j, word in enumerate(vocab):
    mark = " <--" if word in query_tokens else ""
    print(f"    {word:<12} {query_vector[j]:.4f}{mark}")

print(f"\n  {'Doc':<8} cos(Q, Doc)     Result")
print(f"  {'-'*8} -----------     ------")
results = [(i, cosine_similarity(query_vector, v)) for i, v in enumerate(tfidf_vectors)]
best = max(results, key=lambda x: x[1])
for i, sim in results:
    tag = "Most similar" if i == best[0] else ""
    print(f"  Doc{i+1:<5} {sim:.4f}          {tag}")

print(f"\n  Best match: Doc{best[0]+1} -- \"{raw_corpus[best[0]]}\"")

print("\n" + "=" * 55)
print("BONUS — TP website example")
print("=" * 55)
A = [0, 1.2, 1.6]
B = [0, 0.8, 1.4]
dot = sum(a * b for a, b in zip(A, B))
nA = math.sqrt(sum(a**2 for a in A))
nB = math.sqrt(sum(b**2 for b in B))
print(f"  A={A}  B={B}")
print(f"  A.B  = {dot:.4f}")
print(f"  ||A|| = {nA:.4f}   ||B|| = {nB:.4f}")
print(f"  cos  = {dot / (nA * nB):.4f}  (tp gives 0.99) ok")