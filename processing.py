import string
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import unicodedata
from better_profanity import profanity
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def process_reviews(reviews):
    cleaned_reviews = []
    for review in reviews:
        if isinstance(review, str) and review.strip():
            review = unicodedata.normalize('NFC', review)
            review_nopunct = "".join([char for char in review if char not in string.punctuation])
            tokens = re.split(r'\W+', review_nopunct)
            tokens = [word for word in tokens]
            review_processed = [word for word in tokens]
            cleaned_reviews.append(" ".join(review_processed))
        else:
            cleaned_reviews.append("")
    return cleaned_reviews

try:
    # Read data from CSV
    df = pd.read_csv("C:/Users/82nat/OneDrive/Desktop/APEX/reviews_data.csv")

    # Process reviews
    cleaned_reviews = process_reviews(df['Review Text'])

    # Create dataframe with relevant features
    df['Cleaned Review'] = cleaned_reviews
    df['Sentiment'] = df['Cleaned Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Contains Profanity'] = df['Cleaned Review'].apply(lambda x: profanity.contains_profanity(x)).astype(int)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Standardize certain features using StandardScaler
    scaler = StandardScaler()
    df['Capital Letters Standardized'] = scaler.fit_transform(df[['Capital Letters']])
    df['Star Rating Standardized'] = scaler.fit_transform(df[['Star Rating']])
    df['Total Films Reviewed Standardized'] = scaler.fit_transform(df[['Total Films Reviewed']])
    df['Reviews This Year Standardized'] = scaler.fit_transform(df[['Reviews This Year']])
    df['Followers Standardized'] = scaler.fit_transform(df[['Followers']])
    df['Following Standardized'] = scaler.fit_transform(df[['Following']])

    # Calculate and standardize review length
    df['Review Length'] = df['Cleaned Review'].apply(len)
    df['Review Length Standardized'] = scaler.fit_transform(df[['Review Length']])
    df.drop(columns=['Review Length'], inplace=True)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word')
    data_matrix = vectorizer.fit_transform(df['Cleaned Review'])

    # Perform k-means clustering
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data_matrix)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Drop unnecessary columns
    columns_to_drop = ['Capital Letters', 'Star Rating', 'Total Films Reviewed', 'Reviews This Year', 'Following', 'Followers', 'Review Text']
    df.drop(columns=columns_to_drop, inplace=True)

    # Save results to CSV
    df.to_csv("C:/Users/82nat/OneDrive/Desktop/APEX/clustered_reviews.csv", index=False)
    print("Results saved")
except Exception as e:
    print(f"Error: {e}")
