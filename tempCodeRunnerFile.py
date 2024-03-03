def visualize_lof_outliers(data_matrix, lof_outliers):
    pca = PCA(n_components=2)
    numeric_data = data_matrix.select_dtypes(include=[np.number])  # Select only numeric columns
    data_pca = pca.fit_transform(numeric_data)

    print("Length of data_pca:", len(data_pca))
    print("Length of lof_outliers:", len(lof_outliers))
    print("lof_outliers DataFrame:")
    print(lof_outliers)

    colors = plt.cm.tab10(range(2))  # Outliers and inliers are 2 colors

    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c='red', alpha=0.5, label='Outliers')
    plt.scatter(data_pca[lof_outliers.index, 0], data_pca[lof_outliers.index, 1], 
                c='gray', label='Inliers')
    
    plt.title('Outlier Detection using Local Outlier Factor (LOF)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
