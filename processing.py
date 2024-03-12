import string
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
from better_profanity import profanity
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from transformers import BertTokenizer, BertModel
import torch

def process_reviews(reviews):
    cleaned_reviews = []
    for review in reviews:
        if isinstance(review, str) and review.strip():
            review_nopunct = "".join([char for char in review if char not in string.punctuation])
            tokens = re.split(r'\W+', review_nopunct)
            tokens = [word for word in tokens]
            review_processed = [word for word in tokens]
            cleaned_reviews.append(" ".join(review_processed))
        else:
            cleaned_reviews.append("")
    return cleaned_reviews

def count_capital_letters(text):
    return sum(1 for char in text if char.isupper())

try:
    # Read data from CSV
    df = pd.read_csv("C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/reviews_data.csv")
    df.columns = df.columns.astype(str) # Ensure column names are all strings

    # Process reviews
    cleaned_reviews = process_reviews(df['Review Text'])
    df.drop(columns=['Review Text'], inplace=True)

    # Create dataframe with relevant features
    df['Cleaned Review'] = cleaned_reviews
    df['Sentiment'] = df['Cleaned Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Contains Profanity'] = df['Cleaned Review'].apply(lambda x: profanity.contains_profanity(x)).astype(int)
    df['Capital Letters'] = df['Cleaned Review'].apply(count_capital_letters)
    df['Review Length'] = df['Cleaned Review'].apply(len)

    # Standardize all features
    features = ['Sentiment', 'Contains Profanity', 'Capital Letters', 'Star Rating', \
                    'Total Films Reviewed', 'Reviews This Year', 'Followers', 'Following']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Sentence Embedding BERT technique to capture semantic meaning of sentences
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode review text
    encoded_inputs = tokenizer(df['Cleaned Review'].tolist(), padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Extract sentence embeddings from BERT outputs
    # Using the [CLS] token representation for each review
    review_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Concatenate embeddings with other features
    data_matrix = pd.concat([df[features], pd.DataFrame(review_embeddings)], axis=1)

    # Peform TF-IDF Vectorization with L1 Normalization
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', norm='l1')
    data_matrix = vectorizer.fit_transform(df['Cleaned Review'])
    column_names = vectorizer.get_feature_names_out()

    # num_documents, num_dimensions = data_matrix.shape
    # print("Number of documents:", num_documents)
    # print("Number of dimensions aka unique terms:", num_dimensions)

    # Perform SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=10)  
    data_matrix = svd.fit_transform(data_matrix)

    # Concatenate reduced and normalized TF-IDF vectors with the other (standarized) features
    data_matrix = pd.concat([df[features], pd.DataFrame(data_matrix, columns=[f"SVD_{i}" for i in range(svd.n_components)])], axis=1) 

    # Perform k-means clustering
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data_matrix)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Save results to CSV
    df.to_csv("C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/clustered_reviews.csv", index=False)
    print("Results saved")
    
except Exception as e:
    print(f"Error: {e}")
