import nltk
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import pandas as pd
from tqdm import tqdm
from time import perf_counter_ns
from time_utility import print_time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def download_nltk_dependency(dependency_name) -> None:
    """
    Download an NLTK dependency if it's not already available.

    Args:
        dependency_name (str): The name of the NLTK dependency to download.
        language (str): Optional. Specify the language for certain dependencies.

    Returns:
        None
    """
    dependency_identifier = f'{dependency_name}.zip'

    if not nltk.data.find(f'corpora/{dependency_identifier}'):
        print(f"Downloading {dependency_name} from nltk.")
        nltk.download(dependency_name)

nltk_deps = [ 'reuters', 'stopwords', 'wordnet' ]
map(download_nltk_dependency, nltk_deps)

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

def add_to_vocabulary(term):
    """
    Adds a word to the vocabulary if it is not already in it.
    Returns the word in order to be used in a functional programming style.
    """
    vocabulary.add(term)
    return term
    #or this to get rid of numbers, like prices in the text ? size 34824 against 35698
    # if not term.isdigit():
    #     vocabulary.add(term)
    #     return term
    # return ""

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
print(f"doc sample {train_doc[0][:100]}")
print(f"doc category {train_categories[0]}")
print(f"doc id {train_ids[0]}")

print(f"Vocabulary size of the collection: {len(vocabulary)}")


def create_document_term_matrix(preprocessed_corpus, doc_ids, vocabulary) -> (pd.DataFrame, int):
    """
    Create the document-term matrix of the preprocessed corpus.
    columns are the terms of the vocabulary and
    rows are the documents id of the corpus.
    """
    start_time = perf_counter_ns()
    dtm = pd.DataFrame(0, index=train_ids, columns=vocabulary)

    for doc, id in tqdm(zip(preprocessed_corpus, doc_ids), total=len(doc_ids), desc="Creating document-term matrix"):
        term_counts = Counter(doc.split())
        for term, count in term_counts.items():
            dtm.at[id, term] = count

    computation_time = perf_counter_ns() - start_time

    return dtm, computation_time

def create_document_term_matrix_with_scikitlearn(preprocessed_corpus, doc_ids, vocabulary) -> (pd.DataFrame, int):
    """
    Create the document-term matrix of the preprocessed corpus.
    columns are the terms of the vocabulary and
    rows are the documents id of the corpus.
    """
    start_time = perf_counter_ns()
    vectorizer = TfidfVectorizer()
    dtm = vectorizer.fit_transform(preprocessed_corpus)

    computation_time = perf_counter_ns() - start_time

    return dtm, computation_time

doc_term_matrix, time = create_document_term_matrix(train_doc, train_ids, list(vocabulary))

print(f"Computation time: {time} ns.")
print_time(time)

doc_term_matrix, time = create_document_term_matrix_with_scikitlearn(train_doc, train_ids, list(vocabulary))

print(f"Computation time: {time} ns.")
print_time(time)
