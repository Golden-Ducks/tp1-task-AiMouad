number_map = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten"
}

docs = {
    "D0": "  .   Give Me 7 of these",
    "D1": "Today she cooked 4 bourak. Later, she added two chamiyya and 1 pizza.",
    "D2": "Five pizza were ready, but 3 bourak burned!",
    "D3": "We only had 8 chamiyya, no pizza, and one tea.",
    "D4": "Is 6 too much? I ate nine bourak lol."
}

def normalize_text(text):

    text = text.lower()
    
    # Remove punctuation manually
    cleaned = ""
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned += char
        else:
            cleaned += " "
    
    # Split into tokens
    tokens = cleaned.split()
    
    # Convert numbers to words
    normalized_tokens = []
    for token in tokens:
        if token in number_map:
            normalized_tokens.append(number_map[token])
        else:
            normalized_tokens.append(token)
    
    # Join back to string
    return " ".join(normalized_tokens)

for key, value in docs.items():
    print(key + ":")
    print(normalize_text(value))
    print()