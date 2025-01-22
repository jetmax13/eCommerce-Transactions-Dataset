import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from fpdf import FPDF

def create_pdf_report(filename, title, content):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, ln=True)
    
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    
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
    
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    return customers_df, products_df, transactions_df

def generate_eda_report(customers_df, products_df, transactions_df):
    plt.figure(figsize=(20, 15))
    
    # Customer Analysis
    plt.subplot(3, 2, 1)
    customers_df['Region'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Customer Distribution by Region')
    
    plt.subplot(3, 2, 2)
    customers_df['SignupDate'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Customer Signups by Year')
    
    # Product Analysis
    plt.subplot(3, 2, 3)
    sns.histplot(products_df['Price'])
    plt.title('Product Price Distribution')
    
    plt.subplot(3, 2, 4)
    products_df['Category'].value_counts().plot(kind='bar')
    plt.title('Products by Category')
    
    # Transaction Analysis
    plt.subplot(3, 2, 5)
    sns.histplot(transactions_df['TotalValue'])
    plt.title('Transaction Value Distribution')
    
    plt.subplot(3, 2, 6)
    transactions_df['TransactionDate'].dt.month.value_counts().sort_index().plot(kind='line')
    plt.title('Transaction Volume by Month')
    
    plt.tight_layout()
    plt.savefig('Vaibhav_Duggal_EDA_Visualizations.png')
    plt.close()

def generate_business_insights(customers_df, products_df, transactions_df):
    insights = []
    
    # Revenue Analysis
    total_revenue = transactions_df['TotalValue'].sum()
    avg_transaction = transactions_df['TotalValue'].mean()
    insights.append(f"1. Revenue Analysis: Total revenue is ${total_revenue:,.2f} with average transaction value of ${avg_transaction:.2f}. This indicates our pricing strategy and customer spending patterns.")
    
    # Customer Geographic Distribution
    top_region = customers_df['Region'].value_counts().index[0]
    region_percent = (customers_df['Region'].value_counts().values[0] / len(customers_df)) * 100
    insights.append(f"2. Geographic Distribution: {top_region} is our strongest market with {region_percent:.1f}% of customers, suggesting potential for targeted regional marketing campaigns.")
    
    # Product Category Performance
    top_category = products_df['Category'].value_counts().index[0]
    category_revenue = transactions_df.merge(products_df, on='ProductID')
    top_category_revenue = category_revenue.groupby('Category')['TotalValue'].sum().max()
    insights.append(f"3. Product Categories: {top_category} is our leading category with ${top_category_revenue:,.2f} in revenue, indicating strong market demand.")
    
    # Customer Acquisition Trends
    recent_signups = customers_df[customers_df['SignupDate'].dt.year >= 2023]
    growth_rate = (len(recent_signups) / len(customers_df)) * 100
    insights.append(f"4. Customer Growth: {growth_rate:.1f}% of our customer base joined since 2023, showing our recent customer acquisition performance.")
    
    # Purchase Behavior
    repeat_customers = transactions_df['CustomerID'].value_counts()
    repeat_rate = (len(repeat_customers[repeat_customers > 1]) / len(repeat_customers)) * 100
    insights.append(f"5. Customer Loyalty: {repeat_rate:.1f}% of customers made multiple purchases, indicating strong customer retention.")
    
    return insights

def main():
    customers_df, products_df, transactions_df = load_data()

    generate_eda_report(customers_df, products_df, transactions_df)

    insights = generate_business_insights(customers_df, products_df, transactions_df)
    create_pdf_report('Vaibhav_Duggal_EDA.pdf', 
                 'Business Insights Report', 
                 insights)

if __name__ == "__main__":
    main()