import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def visualize_elbow_method(sosd):
    plt.plot(range(1, 15), sosd)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances') 
    plt.show()

def compute_cluster_stats(clustered_df):
    groups = clustered_df.groupby('Cluster')  
    stats = {}

    for label, data in groups:
        avg_sentiment = data['Sentiment'].mean()
        avg_review_length = data['Review Length Standardized'].mean()
        avg_capital_letters = data['Capital Letters'].mean()
        contains_profanity_rate = data['Contains Profanity'].mean()
        avg_star_rating = data['Star Rating'].mean()
        avg_total_films_reviewed = data['Total Films Reviewed'].mean()
        avg_reviews_this_year = data['Reviews This Year'].mean()
        avg_followers = data['Followers'].mean()
        avg_following = data['Following'].mean()

        stats[label] = {
            'Average Sentiment': avg_sentiment,
            'Average Review Length': avg_review_length,
            'Average Capital Letters': avg_capital_letters,
            'Contains Profanity Rate': contains_profanity_rate,
            'Average Star Rating': avg_star_rating,
            'Average Total Films Reviewed': avg_total_films_reviewed,
            'Average Reviews This Year': avg_reviews_this_year,
            'Average Followers': avg_followers,
            'Average Following': avg_following
        }

    for label, statistics in stats.items():
        print(f"Cluster {label} Statistics:")
        for name, value in statistics.items():
            print(f"- {name}: {value:.2f}")
        print()

def compute_silhouette_score(data_matrix, cluster_labels):
    silhouette_avg = silhouette_score(data_matrix, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

def visualize_clusters(data_matrix, cluster_labels):
    pca = PCA(n_components=2) # Reducing the dimensions to 2 so we can visualize it in a graph
    data_pca = pca.fit_transform(data_matrix.toarray())

    colors = plt.cm.tab10(range(cluster_labels.nunique()))

    # So we can see the correct colors
    plt.figure(figsize=(8, 6))
    for cluster_label in range(cluster_labels.nunique()):
        plt.scatter(data_pca[cluster_labels == cluster_label, 0], 
                    data_pca[cluster_labels == cluster_label, 1], 
                    c=[colors[cluster_label]], label=f'Cluster {cluster_label}')

    plt.title('Review Clusters')
    plt.xlabel('Principal Component 1') # Each axis is made up of several components, so this is a standard
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

def main():
    clustered_df = pd.read_csv("C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/clustered_reviews.csv")

    # Remove rows with empty reviews
    clustered_df.dropna(subset=['Cleaned Review'], inplace=True)

    # Redo TF-IDF Vectorization 
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word')
    data_matrix = vectorizer.fit_transform(clustered_df['Cleaned Review'])

    # Elbow Method (for determining how many clusters)
    sosd = [] # Sum of Squared Distances
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data_matrix)
        sosd.append(kmeans.inertia_)

    # Plot Elbow Method
    visualize_elbow_method(sosd)

    # Cluster Stats
    if clustered_df['Cluster'].nunique() > 1:
        compute_cluster_stats(clustered_df)

    # Silhouette Score
    compute_silhouette_score(data_matrix, clustered_df['Cluster'])

    # Graph results using PCA
    visualize_clusters(data_matrix, clustered_df['Cluster'])

if __name__ == "__main__":
    main()