import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import reuters
from nltk.stem import PorterStemmer
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('reuters')
nltk.download('stopwords', 'english')
nltk.download('wordnet')

# initiates global variabels
documents = reuters.fileids()
stopwordsPunctuation = stopwords.words('english') + list(string.punctuation)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# display  the beginning of each file to have an overview of the data
# for doc in documents:
#     raw_data = reuters.raw(doc)
#     print(f"Document {doc}:\n{raw_data[:100]}...\n\n")

# build the vocabulary of the collection, needed to make the document term-matrix
vocabulary = set()

def add_to_vocabulary(word):
    """
    Adds a word to the vocabulary if it is not already in it.
    Returns the word in order to be used in a functional programming style.
    """
    vocabulary.add(word)
    return word

# preproces the data
# Explanations zip(*... :
# we build an array containning : [[doc1, cat1, id1], [doc2, cat2, id2], ...]
# so in order to unpack this into variables doc, cat, id, need to concat the columns of each row
# Thus we first unpack the array using * to obtain n arrays : [doc1, cat1, id1], [doc2, cat2, id2], ...
# and then we use zip to concatenate thos arrays into one array : [[doc1,doc2], [cat1,cat2], [id1,id2], ...]
train_doc, train_categories, train_ids = zip(*[
    [
    ' '.join([
        add_to_vocabulary(stemmer.stem(lemmatizer.lemmatize(w)))
        for w in word_tokenize(reuters.raw(doc_id).lower())
        if not w in stopwordsPunctuation
        ]),
     reuters.categories(doc_id),
     doc_id
    ] 
    for doc_id in reuters.fileids() 
    if doc_id.startswith("train")
])

# display the first document to see the result
print(f"Number of training documents: {len(train_doc)}")
print(train_doc[0])
print(train_categories[0])
print(train_ids[0])

print(f"Vocabulary of the collection {vocabulary}")


def document_term_matrix(preprocessed_corpus):
    M = {}

    # for 

    for doc in preprocessed_corpus:
        for token in doc.split():
            pass