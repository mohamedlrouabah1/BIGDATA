import nltk
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

nltk_deps = [ 'reuters', 'stopwords', 'wordnet', 'punkt' ]
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
    for doc_id in tqdm(reuters.fileids(), desc="Preprocessing documents", colour="green") 
    if doc_id.startswith("train")
])

# display the first document to see the result
print(f"Number of training documents: {len(train_doc)}")
print(f"doc id:\n{train_ids[0]}")
print(f"doc category:\ {train_categories[0]}")
print(f"doc sample:\n {train_doc[0][:100]}")

print(f"Vocabulary size of the collection: {len(vocabulary)}")


def create_document_term_matrix(preprocessed_corpus, doc_ids, vocabulary) -> (pd.DataFrame, int):
    """
    Create the document-term matrix of the preprocessed corpus.
    columns are the terms of the vocabulary and
    rows are the documents id of the corpus.
    """
    start_time = perf_counter_ns()
    dtm = pd.DataFrame(0, index=doc_ids, columns=vocabulary)

    for doc, id in tqdm(zip(preprocessed_corpus, doc_ids), total=len(doc_ids), desc="Creating document-term matrix", colour="red"):
        doc_terms = doc.split()
        total_nb_terms = len(doc_terms) if len(doc_terms) != 0 else 1
        terms_count = Counter(doc_terms)
        
        for term, count in terms_count.items():
            dtm.at[id, term] = count // total_nb_terms

    computation_time = perf_counter_ns() - start_time

    return dtm, computation_time

def create_document_term_matrix_with_scikitlearn(preprocessed_corpus, use_idf=False) -> (pd.DataFrame, int):
    """
    Create the document-term matrix of the preprocessed corpus.
    columns are the terms of the vocabulary and
    rows are the documents id of the corpus.
    """
    start_time = perf_counter_ns()
    vectorizer = TfidfVectorizer(input="content", tokenizer=word_tokenize, stop_words=stopwordsPunctuation, use_idf=use_idf)
    dtm = vectorizer.fit_transform(preprocessed_corpus)

    computation_time = perf_counter_ns() - start_time

    return dtm, vectorizer, computation_time

dtm, time = create_document_term_matrix(train_doc, train_ids, list(vocabulary))

print(f"Computation time: {time} ns.")
print_time(time)

dtm_sklearn, _, time = create_document_term_matrix_with_scikitlearn(train_doc)

print(f"Computation time with scikitlearn: {time} ns.")
print_time(time)

dtm_tfidf_sklearn, tfvectorizer, time = create_document_term_matrix_with_scikitlearn(train_doc, use_idf=True)
print(f"Computation time with scikitlearn and tf-idf: {time} ns.")
print_time(time)


# compute the frequency of the each terms in the collection
term_frequencies = np.array(dtm_tfidf_sklearn.sum(axis=0)).flatten()

# Sort term frequencies in descending order
sorted_term_indices = np.argsort(term_frequencies)[::]
# Note: The np.argsort() function returns the indices that would sort an array,

terms = tfvectorizer.get_feature_names_out()
ranked_term_asc = [terms[i] for i in sorted_term_indices]
print(f"Term frequencies: {ranked_term_asc[:10]}")

# Plot frequency vs. rank
df = pd.DataFrame({'term': terms, 'frequency': term_frequencies})
# Customize X-axis labels (display vertically)
# plt.xticks(df['term'][:20], rotation=90, verticalalignment="center", fontsize=8)
ax = df.plot(x='term', y='frequency', logy=True, figsize=(12, 6))



# Perform LSA using TruncatedSVD:
from sklearn.decomposition import TruncatedSVD

# Specify the number of components (topics) for LSA
num_components = 90

lsa = TruncatedSVD(n_components=num_components)
lsa_result = lsa.fit_transform(dtm_tfidf_sklearn)

#Cluster the terms and documents using your favorite clustering methods. 
# You can use methods like K-means, hierarchical clustering, 
# or any other clustering algorithm of your choic
from sklearn.cluster import KMeans

# Specify the number of clusters for term clustering
num_term_clusters = 90

kmeans = KMeans(n_clusters=num_term_clusters)
term_clusters = kmeans.fit_predict(lsa_result)


# Explore the clusters
def get_top_terms_per_cluster(lsa_result, term_clusters, terms, num_top_terms=10):
    top_terms_per_cluster = []

    for cluster_id in range(num_term_clusters):
        cluster_indices = np.where(term_clusters == cluster_id)[0]
        cluster_term_weights = lsa_result[cluster_indices]  # Get the term weights for the cluster
        cluster_term_weights_sum = cluster_term_weights.sum(axis=0)  # Sum term weights along cluster samples

        # Find the indices of the top terms based on their weights
        top_term_indices = np.argsort(cluster_term_weights_sum)[::-1][:num_top_terms]
        top_cluster_terms = [terms[i] for i in top_term_indices]
        top_terms_per_cluster.append(top_cluster_terms)

    return top_terms_per_cluster

# Get the top terms in each cluster
top_terms_per_cluster = get_top_terms_per_cluster(lsa_result, term_clusters, terms)



# 7 word embedding
from gensim.models import Word2Vec

corpus = train_doc
# Create and train the Word2Vec model
model = Word2Vec(corpus, vector_size=16, window=5, min_count=1, sg=0)

# TODO: adjust hyperparameters like 'window', 'min_count', and 'sg' as needed.

# Get the vector for the word 'computer':
vector = model.wv['computer']

# Find similar words to 'computer' using Word2Vec:
similar_words = model.wv.most_similar('computer', topn=10)
print(similar_words)

# To compare the set of similar words given by Word2Vec to the set of words in the cluster 
# (from your previous section), you can use Python's set operations. Assuming you have a list of 
# similar words from Word2Vec and a list of cluster terms:
# List of similar words from Word2Vec
word2vec_similar_words = ['laptop', 'device', 'machine', 'computers', 'hardware', 'desktop', 'software', 'laptops', 'PC']

# List of cluster terms (adjust this based on your actual cluster results)
cluster_terms = ['machine', 'device', 'laptop', 'hardware', 'software']

# Convert both lists to sets for easy comparison
word2vec_similar_set = set(word2vec_similar_words)
cluster_set = set(cluster_terms)

# Find common words between Word2Vec and cluster terms
common_words = word2vec_similar_set.intersection(cluster_set)

# Find words unique to Word2Vec and cluster terms
unique_to_word2vec = word2vec_similar_set.difference(cluster_set)
unique_to_cluster = cluster_set.difference(word2vec_similar_set)

print("Common Words:", common_words)
print("Words Unique to Word2Vec:", unique_to_word2vec)
print("Words Unique to Cluster:", unique_to_cluster)
