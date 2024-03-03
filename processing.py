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

    # Process reviews
    cleaned_reviews = process_reviews(df['Review Text'])
    df.drop(columns=['Review Text'], inplace=True)

    # Create dataframe with relevant features
    df['Cleaned Review'] = cleaned_reviews
    df['Sentiment'] = df['Cleaned Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Contains Profanity'] = df['Cleaned Review'].apply(lambda x: profanity.contains_profanity(x)).astype(int)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Calculate number of capital letters in review text
    df['Capital Letters'] = df['Cleaned Review'].apply(count_capital_letters)

    # Standardize features 
    scaler = StandardScaler()
    df[['Capital Letters', 'Star Rating', 'Total Films Reviewed', 'Reviews This Year', 'Followers', 'Following']] = \
        scaler.fit_transform(df[['Capital Letters', 'Star Rating', 'Total Films Reviewed', 'Reviews This Year', 'Followers', 'Following']])

    # Calculate and standardize review length
    df['Review Length'] = df['Cleaned Review'].apply(len)
    df['Review Length Standardized'] = scaler.fit_transform(df[['Review Length']])
    # Drop the original 'Review Length' column
    df.drop(columns=['Review Length'], inplace=True)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word')
    data_matrix = vectorizer.fit_transform(df['Cleaned Review'])
    column_names = vectorizer.get_feature_names_out()

    data_matrix = pd.concat([df[['Sentiment', 'Contains Profanity', 'Capital Letters', 'Star Rating', 'Total Films Reviewed', 'Reviews This Year', 'Followers', 'Following']], pd.DataFrame(data_matrix.toarray(), columns=column_names)], axis=1)

    # Standardize all features
    scaler_combined = StandardScaler()
    data_matrix = scaler_combined.fit_transform(data_matrix)

    # Perform k-means clustering
    k = 7
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data_matrix)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Save results to CSV
    df.to_csv("C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/clustered_reviews.csv", index=False)
    print("Results saved")
    
except Exception as e:
    print(f"Error: {e}")
