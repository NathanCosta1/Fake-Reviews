import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def visualize_lof_outliers(data_matrix, lof_outliers):
    pca = PCA(n_components=2)
    numeric_data = data_matrix.select_dtypes(include=[np.number])  # Select only numeric columns
    data_pca = pca.fit_transform(numeric_data)

    lof_outliers_reset = lof_outliers.reset_index(drop=True)  # Reset the index of lof_outliers
    
    colors = plt.cm.tab10(range(2))  # Outliers and inliers are 2 colors

    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c='red', alpha=0.5, label='Outliers')
    plt.scatter(data_pca[lof_outliers_reset.index, 0], data_pca[lof_outliers_reset.index, 1], 
                c='gray', label='Inliers')
    
    plt.title('Outlier Detection using Local Outlier Factor (LOF)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def lof_outlier_detection(data_matrix):
    numeric_columns = data_matrix.select_dtypes(include=[np.number]).columns
    data_matrix_numeric = data_matrix[numeric_columns]

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof.fit_predict(data_matrix_numeric)
    lof_scores = -lof.negative_outlier_factor_  

    threshold = sorted(lof_scores)[int(0.1 * len(lof_scores))]
    lof_outliers = pd.DataFrame({'LOF Score': lof_scores})
    lof_outliers = lof_outliers[lof_outliers['LOF Score'] > threshold]
    
    return lof_outliers

def main():
    clustered_df = pd.read_csv("C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/clustered_reviews.csv")

    # Remove rows with empty reviews
    clustered_df.dropna(subset=['Cleaned Review'], inplace=True)

    # Plot
    lof_outliers = lof_outlier_detection(clustered_df)
    visualize_lof_outliers(clustered_df, lof_outliers)

    # Merge LOF outliers with the main DataFrame based on index
    clustered_df = pd.merge(clustered_df, lof_outliers, left_index=True, right_index=True, how='left', suffixes=('', '_LOF'))

    # Default all values to inlier
    clustered_df['Is Outlier'] = clustered_df['LOF Score'].apply(lambda x: 0 if pd.notnull(x) else 1)

    # Save DataFrame with ONLY LOF score and Is outlier columns
    clustered_df.drop(columns = ['Date', 'Star Rating', 'Total Films Reviewed', 'Reviews This Year', 'Following', 'Followers', 'Sentiment', 'Contains Profanity', 'Capital Letters', 'Review Length Standardized', 'Cluster'], inplace=True )
    clustered_df.to_csv('C:/Users/82nat/OneDrive/Desktop/Career/Current Projects/APEX/lof_outliers.csv', index=False)

if __name__ == "__main__":
    main()
