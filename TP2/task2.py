# Context Window avec Padding 
import re
import numpy as np


D1 = "I love cats"
D2 = "Cats are chill"
D3 = "I am late"


def preprocess(text):
    """Lowercase and tokenize."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()


def add_padding(tokens, window_s=1):
    """Wrap tokens with <s> and </s> flags."""
    return ["<s>"] * window_s + tokens + ["</s>"] * window_s


def extract_windows(tokens, window_s=1):
    """
    Slide a window of size (2*window_size + 1) across padded tokens.
    Returns a list of window strings.
    """
    width = 2 * window_s + 1
    windows = []
    for i in range(len(tokens) - width + 1):
        window = " ".join(tokens[i:i + width])
        windows.append(window)
    return windows


def build_vocab(all_windows):
    """Collect unique windows, sort alphabetically, assign index."""
    unique = sorted(set(w for windows in all_windows for w in windows))
    vocab = {w: i for i, w in enumerate(unique)}
    return vocab


def vectorize_doc(doc_windows, vocab):
    """Return a binary vector: 1 if window present in doc, 0 otherwise."""
    vec = np.zeros(len(vocab), dtype=int)
    for w in doc_windows:
        if w in vocab:
            vec[vocab[w]] = 1
    return vec


# run 
all_docs = [D1, D2, D3]
WINDOW_SIZE = 1

# Step 1 preprocess + pad
padded_docs = [add_padding(preprocess(doc), WINDOW_SIZE) for doc in all_docs]

# Step 2 extract windows per doc
doc_windows = [extract_windows(tokens, WINDOW_SIZE) for tokens in padded_docs]

# Step 3 build global vocab
vocab = build_vocab(doc_windows)

print("Sorted Vocab (Context Window):")
for term, idx in vocab.items():
    print(f"  {idx}: '{term}'")

# Step 4: vectorize each doc
print("\nVectorized Documents:")
for i, (doc, windows) in enumerate(zip(all_docs, doc_windows)):
    vec = vectorize_doc(windows, vocab)
    print(f"  D{i+1} ({doc!r}): {vec.tolist()}")