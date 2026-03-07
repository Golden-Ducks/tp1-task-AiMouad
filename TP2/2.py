import spacy 
import contractions
from num2words import num2words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import re

nlp = spacy.load("en_core_web_sm")

t1 = "Dates are hiding from you today :'))"
t2 = "I won't attend this Sunday evening 13 pm."
t3 = "Cats are watching angels outside :)"

texts_tp2 = [ t1, t2, t3 ]

def number_to_words(text):
    words = text.split()
    new_words = []
    for w in words:
        if w.isdigit():
            new_words.append(num2words(int(w)))
        else:
            new_words.append(w)
    return " ".join(new_words)


def preprocess(text):
    text = contractions.fix(text)
    text = text.lower()
    text = number_to_words(text)
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    return tokens
print("======== Exo 1 :==========")

for t in texts_tp2:
    print(preprocess(t))


documents = ["Muslims are praying and giving charity today",
             "People are fasting and praying during Ramadan",
             "The community gathers to pray and help the needy",
             "Customers are buying and giving feedback today",
             "People are shopping and giving reviews online",
             "The community gathers to buy and help new clients"]

clean_docs = []

for doc in documents:
    processed = preprocess(doc) 
    clean_docs.append(" ".join(processed))
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(clean_docs)

print("\n Vectorized Documents:")
print(vectorizer.get_feature_names_out())

print("\n BOW Matrix:")
print(X.toarray())  

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)


print("\n Cluster :")
print (kmeans.labels_)
