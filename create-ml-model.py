import pandas as pd
import sqlite3
import string
import nltk
import spacy
import sklearn
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

conn = sqlite3.connect("test-courses.db")
df = pd.read_sql_query('SELECT * FROM course_list', conn)
df['college_name'] = df['college_name'].astype(str)
df['course_name'] = df['course_name'].astype(str)
df['course_description'] = df['course_description'].astype(str)

df.drop_duplicates(inplace=True)

def process_text(text):
    custom_stop_words = set([
    'and', 'but', 'for', 'so', 'is', 'the', 'it', 'by', 
    'a', 'an', 'of', 'or', 'to', 'in', 'on', 'with', 'at', 'as', 'course', 'college', 'university', 'stanford'
    ])
    stop_words = set(stopwords.words('english')).union(custom_stop_words)
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    # Remove extra whitespaces
    text = ' '.join(text.split())
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Apply the combined function to the text column
df['processed_text'] = df['course_description'].apply(process_text)
df['processed_text'] = df['processed_text'].apply(lambda x: ' '.join(x))
print(df['processed_text'])
pd.set_option('display.max_columns', None)
test_corpus = df.iloc[:1000]
vectorizer = TfidfVectorizer()
#adjust vectorizer settigns to improve quality of features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, ngram_range=(1, 2))

# Fit and transform the text data in the DataFrame
tfidf_matrix = vectorizer.fit_transform(test_corpus['processed_text'])

# Convert the sparse matrix to a dense matrix (optional)
dense_matrix = tfidf_matrix.toarray()

feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)
print(tfidf_df)


Z = linkage(dense_matrix, method = 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Distance')
plt.show()

cluster_labels = fcluster(Z, t=2.5, criterion='distance')  # 't' is the threshold or the number of clusters
test_corpus['cluster'] = cluster_labels
grouped = test_corpus.groupby('cluster')
tfidf_matrix = vectorizer.fit_transform(test_corpus['processed_text'])
terms = vectorizer.get_feature_names_out()
cluster_summaries = {}
for cluster in grouped.groups:
    # Get indices of items in this cluster
    indices = grouped.groups[cluster]
    
    # Calculate the mean TF-IDF score for each term in the cluster
    mean_tfidf = np.mean(tfidf_matrix[indices].toarray(), axis=0).flatten()

    # Get the top N terms with the highest mean TF-IDF scores
    top_n_terms = np.argsort(mean_tfidf)[::-1][:10]
    top_terms = [terms[i] for i in top_n_terms]
    
    # Store the summary of top terms for each cluster
    cluster_summaries[cluster] = top_terms

# Display the top terms for each cluster
for cluster, terms in cluster_summaries.items():
    print(f"Cluster {cluster}: {', '.join(terms)}")

print('done')
