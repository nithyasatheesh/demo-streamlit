import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load finance product data (replace with your dataset)
data = {
    'ProductID': [1, 2, 3, 4],
    'ProductName': ['Savings Account', 'Investment Fund', 'Credit Card', 'Personal Loan'],
    'Description': [
        'Earn interest on your savings with our high-yield savings account.',
        'Diversify your portfolio with our investment fund options.',
        'Enjoy cashback and rewards with our credit card offers.',
        'Flexible personal loan options tailored to your needs.'
    ]
}

df = pd.DataFrame(data)

# Count Vectorization (simple bag-of-words)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['Description'])

# Compute cosine similarity between products
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Streamlit app
st.title('Finance Product Recommendation App')

# User input for recommendation
user_input = st.text_area('Enter your financial needs:', 'savings and investment')

# Transform user input using Count Vectorizer
user_input_vectorized = count_vectorizer.transform([user_input])

# Compute cosine similarity between user input and product descriptions
cosine_sim_user = cosine_similarity(user_input_vectorized, count_matrix).flatten()

# Get product recommendations
recommendations = df.iloc[cosine_sim_user.argsort()[:-6:-1]]

# Display recommended products
st.header('Top 5 Recommended Finance Products:')
for idx, row in recommendations.iterrows():
    st.write(f'Product: {row["ProductName"]}')
    st.write(f'Description: {row["Description"]}')
    st.write('---')