import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_data():
    customers_df = pd.read_csv('Customers.csv')
    products_df = pd.read_csv('Products.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    return customers_df, products_df, transactions_df

def create_customer_features(customers_df, products_df, transactions_df):
    #Merged data
    merged_data = transactions_df.merge(customers_df, on='CustomerID')
    merged_data = merged_data.merge(products_df, on='ProductID')
    
    # Creating customer features
    customer_features = pd.DataFrame()
    
    # Transaction based features
    customer_features['total_spend'] = merged_data.groupby('CustomerID')['TotalValue'].sum()
    customer_features['avg_transaction_value'] = merged_data.groupby('CustomerID')['TotalValue'].mean()
    customer_features['transaction_count'] = merged_data.groupby('CustomerID')['TransactionID'].count()
    customer_features['unique_products'] = merged_data.groupby('CustomerID')['ProductID'].nunique()
    
    # Category preferences
    category_pivot = pd.pivot_table(
        merged_data, 
        values='TotalValue',
        index='CustomerID',
        columns='Category',
        aggfunc='sum',
        fill_value=0
    )
    customer_features = customer_features.join(category_pivot)
    
    # Region encoding
    region_dummies = pd.get_dummies(customers_df.set_index('CustomerID')['Region'], prefix='region')
    customer_features = customer_features.join(region_dummies)
    
    return customer_features

def find_lookalikes(customer_features, target_customers, n_recommendations=3):

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features)
    
    # Calculating similarity matrix
    similarity_matrix = cosine_similarity(scaled_features)
    
    # Generating recommendations
    recommendations = {}
    for cust_id in target_customers:
        if cust_id in customer_features.index:
            idx = customer_features.index.get_loc(cust_id)
            similar_scores = similarity_matrix[idx]
            similar_indices = np.argsort(similar_scores)[::-1][1:n_recommendations+1]
            
            recommendations[cust_id] = [
                (customer_features.index[i], similar_scores[i])
                for i in similar_indices
            ]
    
    return recommendations

def main():

    
    customers_df, products_df, transactions_df = load_data()

    customer_features = create_customer_features(customers_df, products_df, transactions_df)
    
    # Generating lookalikes 
    target_customers = [f'C{str(i).zfill(4)}' for i in range(1, 21)]
    lookalike_results = find_lookalikes(customer_features, target_customers)
    
    results_data = []
    for cust_id, recommendations in lookalike_results.items():
        row = {
            'CustomerID': cust_id,
            'RecommendedCustomer1': recommendations[0][0],
            'Score1': recommendations[0][1],
            'RecommendedCustomer2': recommendations[1][0],
            'Score2': recommendations[1][1],
            'RecommendedCustomer3': recommendations[2][0],
            'Score3': recommendations[2][1]
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('Vaibhav_Duggal_Lookalike.csv', index=False)

if __name__ == "__main__":
    main()