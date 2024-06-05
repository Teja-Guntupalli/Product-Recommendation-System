import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load data with caching to speed up app loading
@st.cache_data
def load_data():
    users = pd.read_csv("./Dataset/users.csv")
    products = pd.read_csv("./Dataset/products.csv")
    orders = pd.read_csv("./Dataset/orders.csv")
    order_items = pd.read_csv("./Dataset/order_items.csv")
    inventory_items = pd.read_csv('./Dataset/inventory_items.csv')
    return users, products, orders, order_items, inventory_items

users, products, orders, order_items, inventory_items = load_data()

# Preparing the dataset
@st.cache_data
def prepare_data():
    data = pd.merge(users, orders, left_on='id', right_on='user_id',suffixes=('_users','_orders'))
    data = pd.merge(data, order_items, left_on='order_id', right_on='order_id',suffixes=('','_ord_items'))
    data = pd.merge(data, inventory_items, left_on='inventory_item_id', right_on='id',suffixes=('','_inv_items'))
    data = pd.merge(data, products, left_on='product_id', right_on='id',suffixes=('','_prod'))
    # Dropping unnecessary columns
    data = data.drop(columns=['first_name', 'last_name', 'email', 'state', 'street_address', 'postal_code', 'city', 'country', 
                              'latitude', 'longitude', 'traffic_source', 'created_at_users', 'order_id', 'user_id', 'status', 
                              'gender_orders', 'created_at_orders', 'returned_at', 'shipped_at', 'delivered_at', 'num_of_item', 
                              'id_ord_items', 'user_id_ord_items','inventory_item_id', 'status_ord_items', 'created_at',
                               'shipped_at_ord_items', 'delivered_at_ord_items', 'returned_at_ord_items', 'id_inv_items', 
                               'product_id_inv_items', 'created_at_inv_items', 'sold_at', 'cost', 'product_category', 
                               'product_retail_price','product_sku', 'product_distribution_center_id', 'id_prod', 'cost_prod', 
                               'category', 'name', 'brand', 'retail_price', 'department', 'sku', 'distribution_center_id'])
    data.rename(columns={'id': 'user_id', 'gender_users': 'gender'}, inplace=True)
    return data

data = prepare_data()
# Prepare the interaction data
interaction_data = data.groupby(['user_id', 'product_id']).size().reset_index(name='interaction_count')

dataset = Dataset()
dataset.fit((row['user_id'] for index, row in interaction_data.iterrows()),
            (row['product_id'] for index, row in interaction_data.iterrows()))

(interactions, weights) = dataset.build_interactions(((row['user_id'], row['product_id']) for index, row in interaction_data.iterrows()))

@st.cache_data
def load_model():
    with open('pre_train.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model
model = load_model()

# Your provided functions here
def get_preferred_brand(user_id):
    purchased_products = order_items[order_items['user_id']==user_id]['product_id'].values
    purchased_brands = {}
    for i in purchased_products:
        brand = products[products['id']==i]['brand'].values[0]
        if brand in purchased_brands:
            purchased_brands[brand] += 1
        else:
            purchased_brands[brand] = 1
    max_value = max(purchased_brands.values())
    max_keys = [key for key, value in purchased_brands.items() if value == max_value]
    return random.choice(max_keys)

def get_user_age_and_spending(user_id):
    personal_records = data[data['user_id']==user_id]
    age = personal_records['age'].values[0]
    gender = personal_records['gender'].values[0]
    avg_spending = personal_records['sale_price'].sum()/personal_records['sale_price'].count()
    return age, gender, avg_spending

def get_returned_products(user_id):
    return order_items[(order_items['user_id']==user_id) & (order_items['status'].isin(['Cancelled', 'Returned']))]['product_id'].values

def get_best_sellers(num_recommendations=5):
    total_sales_per_product = order_items.groupby('product_id').size()
    top_selling_products = total_sales_per_product.sort_values(ascending=False).head(num_recommendations)
    return top_selling_products.index.tolist()

def get_new_arrivals(num_recommendations=5):
    inventory_data_sorted = inventory_items.sort_values(by='created_at', ascending=False)
    return inventory_data_sorted.head(num_recommendations)['product_id'].tolist()

def get_product_gender_preference(product_id):
    gender = inventory_items[inventory_items['product_id']==product_id]['product_department'].values[0]
    return 'M' if gender=='Men' else 'F'

def get_product_brand(product_id):
    return inventory_items[inventory_items['product_id']==product_id]['product_brand'].values[0]

def get_purchased_products(user_id):
    pur_prod = data[data.user_id == user_id]
    return pur_prod[['product_name', 'product_id', 'sale_price', 'product_brand']]  # Ensure this is a DataFrame

def get_product_price(product_id):
    return data[data['product_id']==product_id]['sale_price'].values[0]

def sample_recommendation(model, dataset, user_ids, num_recommendations=5, max_preferred_brand=3, spending_range=(0.8, 1.2)):
    n_users, n_items = dataset.interactions_shape()
    recommend, output = [], []
    product_id_mapping = list(dataset.mapping()[2].keys())  # Prepare product ID mapping once

    for user_id in user_ids:
        if user_id in users['id'].values.flatten():
            user_index = dataset.mapping()[0][user_id]
            scores = model.predict(user_index, np.arange(n_items))
            preferred_brand = get_preferred_brand(user_id)
            _, gender, avg_spending = get_user_age_and_spending(user_id)  # Assuming gender is correctly retrieved here
            returned_products = set(get_returned_products(user_id))
            purchased_products = set(get_purchased_products(user_id))
            
            # Map scores to product IDs, excluding returned and purchased products
            product_scores = {product_id_mapping[i]: scores[i] for i in range(n_items) if i not in returned_products and i not in purchased_products}
            output.append(product_scores)  # Store product scores for output

            # Create a sorted list of products based on scores, filtering by gender
            valid_products = [
                (prod, score) for prod, score in product_scores.items()
                if prod not in purchased_products and get_product_gender_preference(prod) == gender
            ]
            valid_products.sort(key=lambda x: x[1], reverse=True)

            brand_recommendations = []
            other_recommendations = []

            for prod, score in valid_products:
                if len(brand_recommendations) + len(other_recommendations) == num_recommendations:
                    break
                product_price = get_product_price(prod)

                if 1 or (avg_spending * spending_range[0] <= product_price <= avg_spending * spending_range[1]):
                    if get_product_brand(prod) == preferred_brand and len(brand_recommendations) < max_preferred_brand:
                        brand_recommendations.append(prod)
                    elif len(other_recommendations) < (num_recommendations - max_preferred_brand):
                        other_recommendations.append(prod)
            
            # If there are still slots left to fill, add more from other recommendations
            if len(brand_recommendations) + len(other_recommendations) < num_recommendations:
                fill_count = num_recommendations - (len(brand_recommendations) + len(other_recommendations))
                additional_products = [prod for prod, _ in valid_products if prod not in brand_recommendations and prod not in other_recommendations][:fill_count]
                other_recommendations.extend(additional_products)

            final_recommendations = brand_recommendations + other_recommendations
            print(f"Recommended items for user {user_id}: {final_recommendations}")
            recommend.append(final_recommendations)
        else:
            new_arrived = get_new_arrivals(num_recommendations//2+1)
            best_seller = get_best_sellers(num_recommendations//2+1)
            final_recommendations =  new_arrived + best_seller
            new_user_recommendation = np.random.choice(final_recommendations, size=num_recommendations, replace=False).tolist()
            print(f"Recommended items for user {user_id}: {new_user_recommendation}")
            recommend.append(new_user_recommendation)

    return recommend, output

# Function to vectorize product descriptions and compute similarity
@st.cache_resource
def get_vectorizer_and_matrix(descriptions):
    descriptions = descriptions.fillna("missing").astype(str)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix

# Prepare vectorizer and TF-IDF matrix
vectorizer, tfidf_matrix = get_vectorizer_and_matrix(products['name'])

def search_products(query, vectorizer, tfidf_matrix):
    query = str(query)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 results
    return products.iloc[similar_indices]

# Streamlit UI starts here
st.title('Product Recommendation System')

# Streamlit UI code for product search
st.sidebar.header("Product Search")
search_query = st.sidebar.text_input("Enter keywords to search for products:", "")

if search_query:
    with st.spinner("Searching for products..."):
        result = search_products(search_query, vectorizer, tfidf_matrix)
        if not result.empty:
            st.write("### Search Results:")
            for _, row in result.iterrows():
                st.write(f"**{row['name']}**")
        else:
            st.write("No products found matching your search.")

# Sidebar for user input
with st.sidebar:
    st.header("User Input")
    # Add a radio button to select user type
    user_type = st.radio("Select User Type:", ('Existing User', 'New User'))
    
    # Conditionally display the User ID input
    if user_type == 'Existing User':
        user_id_input = st.text_input("Enter User ID:")
    else:
        user_id_input = None  # Set to None if New User is selected

# Logic to process inputs based on user type
if user_type == 'Existing User' and user_id_input:
    if user_id_input.isdigit():
        user_id = int(user_id_input)
        user_ids = [user_id]

        with st.spinner('Calculating Recommendations...'):
            recommendations, _ = sample_recommendation(model, dataset, user_ids)

        st.subheader("Customer Details")
        if user_id in users['id'].values:
            user_details = users[users['id'] == user_id]
            customer_name = f"{user_details['first_name'].values[0]} {user_details['last_name'].values[0]}"
            st.write("Customer Name:", customer_name)
            age, gender, avg_spending = get_user_age_and_spending(user_id)
            st.write("Age:", age, "Gender:", gender, "Average Spending: $", f"{avg_spending:.2f}")
            st.write("Preferred Brand:", get_preferred_brand(user_id).title())
            st.write("---")

            purchased_products_df = get_purchased_products(user_id)
            if not purchased_products_df.empty:
                st.subheader("Bought Products:")
                purchased_products_df['Brand'] = purchased_products_df['product_brand'].apply(lambda x: x.title() if isinstance(x, str) else None)
                purchased_products_df['Sale Price'] = purchased_products_df['sale_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                purchased_products_df = purchased_products_df.drop(columns=['product_brand', 'sale_price'])
                purchased_products_df.reset_index(drop=True, inplace=True)
                purchased_products_df = purchased_products_df.rename(columns={'product_name': 'Product Name', 'product_id':'ID','Brand': 'Product Brand', 'Sale Price': 'Price'})
                st.dataframe(purchased_products_df)
            else:
                st.write("No products bought yet.")
        else:
            st.error("Invalid User ID. Please enter a valid User ID.")

        st.subheader("Recommendations")
        for user_id, user_recommendations in zip(user_ids, recommendations):
            st.write(f'Recommended Products for User ID: {user_id}')
            product_details_list = []

            for product_id in user_recommendations:
                product_info = products[products['id']==product_id]
                product_data = {
                    'Name': product_info['name'].values[0],
                    'Retail Price': product_info['retail_price'].values[0].round(2),
                    'Brand': get_product_brand(product_id),
                    'Gender Preference': get_product_gender_preference(product_id)
                }
                product_details_list.append(product_data)

            recommended_products_df = pd.DataFrame(product_details_list)
            st.write(recommended_products_df)
    else:
        st.error("User ID should contain digits only. Please enter a valid User ID.")
elif user_type == 'New User':
    with st.spinner('Generating Recommendations for New Users...'):
        user_ids = [34985734]
        new_user_recommendations, _ = sample_recommendation(model, dataset, user_ids, num_recommendations=5)
        st.subheader("Recommendations for New User")
        if new_user_recommendations:
            for user_id, user_recommendations in zip(user_ids, new_user_recommendations):
                product_details_list = []

                for product_id in user_recommendations:
                    product_info = products[products['id']==product_id]
                    product_data = {
                        'Name': product_info['name'].values[0],
                        'Retail Price': product_info['retail_price'].values[0].round(2),
                        'Brand': get_product_brand(product_id),
                        'Gender Preference': get_product_gender_preference(product_id)
                    }
                    product_details_list.append(product_data)

            recommended_products_df = pd.DataFrame(product_details_list)
            st.write(recommended_products_df)
        else:
            st.write("No recommendations available at the moment.")
        
else:
    st.info("Please select a user type to proceed.")

