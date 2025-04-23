import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit page
st.set_page_config(layout="wide", page_title="Airbnb Recommendation System")
st.title("Airbnb Content-Based Recommendation System")

# Load data function
@st.cache_data
def load_data():
    try:
        # Load the actual datasets
        listings = pd.read_csv("data/listings.csv")
        reviews = pd.read_csv("data/reviews.csv", parse_dates=['date'])
        calendar = pd.read_csv("data/calendar.csv", parse_dates=['date'])
        return listings, reviews, calendar
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Clean price function
def clean_price(price):
    if pd.isna(price):
        return np.nan
    return float(re.sub(r'[^\d.]', '', str(price)))

# Process amenities
def process_amenities(amenities_str):
    if pd.isna(amenities_str):
        return []
    # Remove brackets and quotes, split by commas
    amenities_str = amenities_str.strip('{}')
    # Handle quoted strings with commas inside them
    amenities = []
    in_quotes = False
    current = ''
    
    for char in amenities_str:
        if char == '"' and not in_quotes:
            in_quotes = True
        elif char == '"' and in_quotes:
            in_quotes = False
        elif char == ',' and not in_quotes:
            amenities.append(current.strip('"').strip())
            current = ''
        else:
            current += char
    
    if current:
        amenities.append(current.strip('"').strip())
    
    return amenities

def create_feature_matrix(df):
    df['combined_features'] = df['name'].fillna('') + ' ' + \
                              df['description'].fillna('') + ' ' + \
                              df['neighbourhood_cleansed'].fillna('') + ' ' + \
                              df['property_type'].fillna('') + ' ' + \
                              df['room_type'].fillna('')
    
    # Add amenities to combined features
    df['amenities_list'] = df['amenities'].apply(process_amenities)
    df['combined_features'] = df['combined_features'] + ' ' + df['amenities_list'].apply(lambda x: ' '.join(x))
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    return tfidf_matrix

# Get recommendations
def get_recommendations(listing_id, cosine_sim, df, top_n=5):
    # Get the index of the listing
    idx = df.index[df['id'] == listing_id].tolist()[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort listings based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N most similar listings (excluding the listing itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the listing indices
    listing_indices = [i[0] for i in sim_scores]
    
    # Return the top N similar listings with similarity scores
    recommendations = df.iloc[listing_indices][['id', 'name', 'property_type', 'room_type', 'price', 'review_scores_rating']].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    return recommendations

# Main application
def main():
    # Add a sidebar for app options
    st.sidebar.header("Options")
    
    # Load data
    with st.spinner("Loading data..."):
        listings, reviews, calendar = load_data()
    
    if listings is None:
        st.error("Failed to load data. Please check that your dataset files are in the correct location.")
        return
    
    # Clean the price column
    listings['price_float'] = listings['price'].apply(clean_price)
    
    # Data Processing for Recommendations
    with st.spinner("Processing data for recommendations..."):
        # Create feature matrix
        tfidf_matrix = create_feature_matrix(listings)
        
        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Show some basic stats
    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.text(f"Total Listings: {len(listings)}")
    st.sidebar.text(f"Total Reviews: {len(reviews)}")
    st.sidebar.text(f"Average Rating: {listings['review_scores_rating'].mean():.2f}/100")
    
    # Filtering options
    st.sidebar.subheader("Filter Listings")
    
    # Filter by neighborhood
    neighborhoods = ['All'] + sorted(listings['neighbourhood_cleansed'].dropna().unique().tolist())
    selected_neighborhood = st.sidebar.selectbox("Select Neighborhood", neighborhoods)
    
    # Filter by property type
    property_types = ['All'] + sorted(listings['property_type'].dropna().unique().tolist())
    selected_property = st.sidebar.selectbox("Select Property Type", property_types)
    
    # Filter by price range
    min_price = int(listings['price_float'].min())
    max_price = int(listings['price_float'].max())
    price_range = st.sidebar.slider("Price Range ($)", min_price, max_price, (min_price, max_price))
    
    # Filter by number of bedrooms
    bedrooms = ['All'] + sorted(listings['bedrooms'].dropna().unique().astype(int).tolist())
    selected_bedrooms = st.sidebar.selectbox("Bedrooms", bedrooms)
    
    # Apply filters
    filtered_listings = listings.copy()
    
    if selected_neighborhood != 'All':
        filtered_listings = filtered_listings[filtered_listings['neighbourhood_cleansed'] == selected_neighborhood]
    
    if selected_property != 'All':
        filtered_listings = filtered_listings[filtered_listings['property_type'] == selected_property]
    
    filtered_listings = filtered_listings[(filtered_listings['price_float'] >= price_range[0]) & 
                                         (filtered_listings['price_float'] <= price_range[1])]
    
    if selected_bedrooms != 'All':
        filtered_listings = filtered_listings[filtered_listings['bedrooms'] == selected_bedrooms]
    
    # Main content area
    st.header("Find Similar Airbnb Listings")
    
    # Display filtered listings as a table
    st.subheader("Filtered Listings")
    
    if len(filtered_listings) > 0:
        # Select columns to display
        display_cols = ['id', 'name', 'property_type', 'room_type', 'price', 'bedrooms', 'bathrooms', 'review_scores_rating']
        st.dataframe(filtered_listings[display_cols].head(50))
        
        # Select a listing for recommendations
        listing_ids = filtered_listings['id'].tolist()
        selected_id = st.selectbox("Select a listing ID to find similar properties:", listing_ids)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding similar listings..."):
                # Get recommendations
                recommendations = get_recommendations(selected_id, cosine_sim, listings, top_n=num_recommendations)
                
                # Show the selected listing
                st.subheader("Selected Listing")
                selected_listing = listings[listings['id'] == selected_id].iloc[0]
                st.write(f"**Name:** {selected_listing['name']}")
                st.write(f"**Type:** {selected_listing['property_type']} - {selected_listing['room_type']}")
                st.write(f"**Price:** {selected_listing['price']}")
                st.write(f"**Location:** {selected_listing['neighbourhood_cleansed']}")
                st.write(f"**Rating:** {selected_listing['review_scores_rating'] if not pd.isna(selected_listing['review_scores_rating']) else 'No ratings yet'}")
                
                # Show recommendations
                st.subheader("Recommended Listings")
                for _, row in recommendations.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{row['name']}**")
                        st.write(f"Type: {row['property_type']} - {row['room_type']}")
                        st.write(f"Price: {row['price']}")
                        st.write(f"Rating: {row['review_scores_rating'] if not pd.isna(row['review_scores_rating']) else 'No ratings yet'}")
                    with col2:
                        st.write(f"Similarity Score: {row['similarity_score']:.2f}")
                    st.divider()
    else:
        st.warning("No listings match your filters. Please adjust your criteria.")
    
    # Add a section for visualization
    st.header("Data Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Ratings Analysis", "Neighborhood Comparison"])
    
    with tab1:
        st.subheader("Price Distribution by Property Type")
        # Create a figure for price distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top 5 property types by count
        top_property_types = listings['property_type'].value_counts().head(5).index.tolist()
        plot_data = listings[listings['property_type'].isin(top_property_types)]
        
        # Create boxplot
        sns.boxplot(x='property_type', y='price_float', data=plot_data, ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel('Price ($)')
        plt.xlabel('Property Type')
        plt.title('Price Distribution by Property Type')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Rating Distribution")
        # Create histogram of ratings
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(listings['review_scores_rating'].dropna(), bins=20, kde=True, ax=ax)
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Distribution of Ratings')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Average Price by Neighborhood")
        # Get top 10 neighborhoods by count
        top_neighborhoods = listings['neighbourhood_cleansed'].value_counts().head(10).index.tolist()
        neighborhood_data = listings[listings['neighbourhood_cleansed'].isin(top_neighborhoods)]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        neighborhood_avg = neighborhood_data.groupby('neighbourhood_cleansed')['price_float'].mean().sort_values(ascending=False)
        sns.barplot(x=neighborhood_avg.index, y=neighborhood_avg.values, ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel('Average Price ($)')
        plt.xlabel('Neighborhood')
        plt.title('Average Price by Neighborhood')
        st.pyplot(fig)

if __name__ == "__main__":
    main()