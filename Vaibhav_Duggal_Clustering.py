import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from fpdf import FPDF

def create_pdf_report(filename, title, content):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, title, ln=True)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 9)
    
    if isinstance(content, list):
        for item in content:
            pdf.multi_cell(0, 10, item)
            pdf.ln(5)
    else:
        pdf.multi_cell(0, 10, str(content))
    
    pdf.output(filename)

def load_data():
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    return customers_df, products_df, transactions_df

def create_customer_features(customers_df, products_df, transactions_df):
    merged_data = transactions_df.merge(customers_df, on='CustomerID')
    merged_data = merged_data.merge(products_df, on='ProductID')

    customer_features = pd.DataFrame()

    customer_features['total_spend'] = merged_data.groupby('CustomerID')['TotalValue'].sum()
    customer_features['avg_transaction_value'] = merged_data.groupby('CustomerID')['TotalValue'].mean()
    customer_features['transaction_count'] = merged_data.groupby('CustomerID')['TransactionID'].count()
    customer_features['unique_products'] = merged_data.groupby('CustomerID')['ProductID'].nunique()
    
    category_pivot = pd.pivot_table(
        merged_data, 
        values='TotalValue',
        index='CustomerID',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    )
    customer_features = customer_features.join(category_pivot)
    
    return customer_features

def find_optimal_clusters(scaled_features):
    db_scores = []
    n_clusters_range = range(2, 11)
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        db_score = davies_bouldin_score(scaled_features, labels)
        db_scores.append(db_score)
    
    optimal_n_clusters = n_clusters_range[np.argmin(db_scores)]
    return optimal_n_clusters, db_scores

def perform_clustering(customer_features):

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features)
    
    # Finding optimal number of clusters
    optimal_n_clusters, db_scores = find_optimal_clusters(scaled_features)
    
    # Performing final clustering
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    
    # Creating visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=labels, cmap='viridis')
    plt.title('Customer Segments Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter)
    plt.savefig('Vaibhav_Duggal_Clustering_Visualization.png')
    plt.close()
    
    # Analyzing clusters
    customer_features['Cluster'] = labels
    cluster_stats = customer_features.groupby('Cluster').mean()
    
    return {
        'optimal_clusters': optimal_n_clusters,
        'db_scores': db_scores,
        'labels': labels,
        'cluster_stats': cluster_stats
    }

def main():
    customers_df, products_df, transactions_df = load_data()
    customer_features = create_customer_features(customers_df, products_df, transactions_df)
    
    clustering_results = perform_clustering(customer_features)
    
    content = [
    f"Number of Optimal Clusters: {clustering_results['optimal_clusters']}\n",
    "Davies-Bouldin Index Scores:",
    *[f"Clusters {n_clusters-2}: {score:.4f}" 
      for n_clusters, score in enumerate(clustering_results['db_scores'], 2)],
    "\nCluster Statistics:",
    str(clustering_results['cluster_stats'])
]
    create_pdf_report('Vaibhav_Duggal_Clustering.pdf',
                 'Customer Segmentation Results',
                 content)

if __name__ == "__main__":
    main()