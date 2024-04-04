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
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def process_reviews(reviews):
    cleaned_reviews = []
    for review in reviews:
        if isinstance(review, str) and review.strip():
            review_nopunct = "".join([char for char in review if char not in string.punctuation])
            tokens = re.split(r'\W+', review_nopunct)
            tokens = [word for word in tokens]
            review_processed = [word for word in tokens if word.lower() not in stopwords]
            cleaned_reviews.append(" ".join(review_processed))
        else:
            cleaned_reviews.append("")
    return cleaned_reviews

def count_capital_letters(text):
    return sum(1 for char in text if char.isupper())

try:
    output_file = "C:/Users/82nat/Desktop/490/clustered_reviews_TEST.csv"

    chunk_size = 2000  # Batch size for processing data

    for chunk_df in pd.read_csv("C:/Users/82nat/Desktop/490/reviews_data.csv", chunksize=chunk_size):
        chunk_df.columns = chunk_df.columns.astype(str)  # Ensure column names are all strings

        # Process reviews
        cleaned_reviews = process_reviews(chunk_df['Review Text'])
        chunk_df.drop(columns=['Review Text'], inplace=True)

        # Create dataframe with relevant features
        chunk_df['Cleaned Review'] = cleaned_reviews
        chunk_df['Sentiment'] = chunk_df['Cleaned Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
        chunk_df['Contains Profanity'] = chunk_df['Cleaned Review'].apply(lambda x: profanity.contains_profanity(x)).astype(int)
        chunk_df['Capital Letters'] = chunk_df['Cleaned Review'].apply(count_capital_letters)
        chunk_df['Review Length'] = chunk_df['Cleaned Review'].apply(len)

        # Standardize all features
        features = ['Sentiment', 'Contains Profanity', 'Capital Letters', 'Star Rating', \
                        'Total Films Reviewed', 'Reviews This Year', 'Followers', 'Following']
        scaler = StandardScaler()
        chunk_df[features] = scaler.fit_transform(chunk_df[features])

        vectorizer = TfidfVectorizer()  # You can adjust parameters as needed
        tfidf_vectors = vectorizer.fit_transform(cleaned_reviews)

       # Perform SVD for dimensionality reduction on TFIDF vectors
        svd_tfidf = TruncatedSVD(n_components=10)  
        tfidf_vectors_reduced = svd_tfidf.fit_transform(tfidf_vectors)

        # Create DataFrame with reduced TFIDF vectors
        tfidf_vectors_reduced_df = pd.DataFrame(tfidf_vectors_reduced, columns=[f"SVD_TFIDF_{i}" for i in range(svd_tfidf.n_components)])

        # Tokenize reviews using BERT tokenizer
        tokenized_reviews = [tokenizer.encode(review, add_special_tokens=True, truncation=True, max_length=512) for review in chunk_df['Cleaned Review']]

        with torch.no_grad():
            bert_outputs = [model(torch.tensor([review])).last_hidden_state.mean(dim=1).numpy() for review in tokenized_reviews]

       # Perform SVD for dimensionality reduction on BERT embeddings
        svd_bert = TruncatedSVD(n_components=10, algorithm='randomized')
        bert_embeddings_reduced = svd_bert.fit_transform(np.concatenate(bert_outputs, axis=0))

        # Create DataFrame with reduced BERT embeddings
        bert_embeddings_reduced_df = pd.DataFrame(bert_embeddings_reduced, columns=[f"SVD_BERT_{i}" for i in range(svd_bert.n_components)])

        # Concatenate reduced BERT embeddings with reduced TFIDF vectors and other features
        data_matrix = pd.concat([chunk_df[features], bert_embeddings_reduced_df, tfidf_vectors_reduced_df], axis=1)

        # Drop rows with NaN values
        data_matrix.dropna(inplace=True)

        if not data_matrix.empty:
            # Perform k-means clustering
            k = 4
            kmeans = KMeans(n_clusters=k, random_state=5)
            kmeans.fit(data_matrix)

            # Add cluster labels to the DataFrame
            chunk_df['Cluster'] = kmeans.labels_

            chunk_df = pd.concat([chunk_df, bert_embeddings_reduced_df], axis=1)
            chunk_df = pd.concat([chunk_df, tfidf_vectors_reduced_df], axis=1)

            # Save results to CSV
            mode = 'w' if chunk_df.index[0] == 0 else 'a'
            chunk_df.to_csv(output_file, mode=mode, index=False, header=mode=='w')

        print("Results saved")


except Exception as e:
    print(f"Error: {e}")
