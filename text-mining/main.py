import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import reuters
from nltk.stem import PorterStemmer
import string
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

# preproces the data
train_doc, train_categories, train_ids = zip(*[
    [
    ' '.join([
        stemmer.stem(lemmatizer.lemmatize(w))
        for w in word_tokenize(reuters.raw(doc_id).lower())
        if not w in stopwordsPunctuation
        ]),
     reuters.categories(doc_id),
     doc_id
    ] 
    for doc_id in reuters.fileids() if doc_id.startswith("train")
])

# display the first document to see the result
print(f"Number of training documents: {len(train_doc)}")
print(train_doc[0])
print(train_categories[0])
print(train_ids[0])

def document_term_matrix(preprocessed_corpus):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(preprocessed_corpus), vectorizer

docuemnt_term_matrix, vectorizer = document_term_matrix(train_doc)

print(docuemnt_term_matrix.shape)
print(docuemnt_term_matrix[0])
print(vectorizer.get_feature_names_out()[0:10])