import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def visualize_lof_outliers(data_matrix, lof_outliers):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_matrix.toarray())

    colors = plt.cm.tab10(range(2))  # Assuming outliers and inliers are two colors

    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c='red', alpha=0.5, label='Outliers')
    plt.scatter(data_pca[lof_outliers.index, 0], data_pca[lof_outliers.index, 1], 
                c='gray', label='Inliers')
    
    plt.title('Outlier Detection using Local Outlier Factor (LOF)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def lof_outlier_detection(data_matrix):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)  # Can change to make outliers more or less sensitive
    lof.fit_predict(data_matrix)
    lof_scores = -lof.negative_outlier_factor_  
    
    threshold = sorted(lof_scores)[int(0.1 * len(lof_scores))]  # Can change to make outliers more or less sensitive
    lof_outliers = pd.DataFrame({'LOF Score': lof_scores})
    lof_outliers = lof_outliers[lof_outliers['LOF Score'] > threshold]
    
    return lof_outliers

def main():
    clustered_df = pd.read_csv("C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/clustered_reviews.csv")

    # Remove rows with empty reviews
    clustered_df.dropna(subset=['Cleaned Review'], inplace=True)

    # Redo TF-IDF Vectorization 
    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word')
    data_matrix = vectorizer.fit_transform(clustered_df['Cleaned Review'])

    # Plot and save LOF outliers
    lof_outliers = lof_outlier_detection(data_matrix)
    # visualize_lof_outliers(data_matrix, lof_outliers)
        
    # Reset index of lof_outliers to ensure alignment with clustered_df
    lof_outliers.reset_index(drop=True, inplace=True)

    # Add column indicating inliers (1) and outliers (0)
    lof_outliers['Is Outlier'] = 1  # Default to outliers
    # Update the 'Is Outlier' column for rows identified as inliers
    clustered_df.loc[lof_outliers.index, 'Is Outlier'] = 0

    # # Add Review text for manual review
    lof_outliers = pd.merge(lof_outliers, clustered_df[['Cleaned Review']], left_index=True, right_index=True)

    lof_outliers.to_csv('C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/lof_outliers.csv', index=False)

if __name__ == "__main__":
    main()
