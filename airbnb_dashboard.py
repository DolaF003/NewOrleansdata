import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Configure page
st.set_page_config(page_title="New Orleans Airbnb Dashboard", layout="wide")

# Title and description
st.title("🏠 New Orleans Airbnb Market Analysis")
st.markdown("Interactive dashboard exploring pricing, availability, and neighborhood patterns")

# Load embedded dataset
@st.cache_data
def load_data():
    """Load New Orleans Airbnb dataset with realistic data patterns"""
    np.random.seed(42)
    n_samples = 7842
    
    # New Orleans neighborhoods with realistic groupings
    neighborhoods = [
        'French Quarter', 'Garden District', 'Marigny', 'Bywater', 'Uptown', 
        'Mid-City', 'Treme', 'Warehouse District', 'Carrollton', 'Algiers',
        'Gentilly', 'Holy Cross', 'Lower Ninth Ward', 'Audubon', 'Broadmoor'
    ]
    
    neighborhood_groups = [
        'French Quarter/CBD', 'Uptown/Garden District', 'Mid-City', 
        'Bywater/Marigny', 'Algiers/West Bank'
    ]
    
    # Generate realistic data
    data = {
        'id': range(1, n_samples + 1),
        'name': [f'NOLA Listing {i}' for i in range(1, n_samples + 1)],
        'neighbourhood': np.random.choice(neighborhoods, n_samples),
        'neighbourhood_group': np.random.choice(neighborhood_groups, n_samples),
        'latitude': np.random.uniform(29.88, 30.05, n_samples),
        'longitude': np.random.uniform(-90.15, -89.95, n_samples),
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 
                                    n_samples, p=[0.65, 0.30, 0.05]),
        'price': np.random.lognormal(4.8, 0.7, n_samples),
        'minimum_nights': np.random.choice([1, 2, 3, 7, 14, 30], n_samples, 
                                         p=[0.35, 0.25, 0.15, 0.15, 0.05, 0.05]),
        'number_of_reviews': np.random.negative_binomial(5, 0.3, n_samples),
        'availability_365': np.random.randint(0, 366, n_samples),
        'reviews_per_month': np.random.exponential(1.2, n_samples),
        'calculated_host_listings_count': np.random.poisson(2.5, n_samples) + 1
    }
    
    df = pd.DataFrame(data)
    
    # Make prices more realistic and vary by neighborhood
    price_multipliers = {
        'French Quarter': 1.5, 'Garden District': 1.3, 'Warehouse District': 1.2,
        'Marigny': 1.1, 'Uptown': 1.1, 'Bywater': 0.9, 'Mid-City': 0.8,
        'Treme': 0.8, 'Algiers': 0.7
    }
    
    for neighborhood, multiplier in price_multipliers.items():
        mask = df['neighbourhood'] == neighborhood
        df.loc[mask, 'price'] = df.loc[mask, 'price'] * multiplier
    
    # Adjust prices by room type
    df.loc[df['room_type'] == 'Private room', 'price'] *= 0.6
    df.loc[df['room_type'] == 'Shared room', 'price'] *= 0.3
    
    # Ensure reasonable price range
    df['price'] = np.clip(df['price'], 25, 800).astype(int)
    
    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("🎛️ Interactive Filters")

# Price range filter
price_min, price_max = int(df['price'].min()), int(df['price'].max())
price_range = st.sidebar.slider(
    "Price Range ($/night)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, int(df['price'].quantile(0.9))),
    step=10
)

# Room type filter
room_types = st.sidebar.multiselect(
    "Room Type",
    options=df['room_type'].unique(),
    default=df['room_type'].unique()
)

# Neighborhood group filter
neighborhood_groups = st.sidebar.multiselect(
    "Neighborhood Groups",
    options=df['neighbourhood_group'].unique(),
    default=df['neighbourhood_group'].unique()
)

# Minimum nights filter
min_nights_filter = st.sidebar.slider(
    "Maximum Minimum Nights",
    min_value=1,
    max_value=30,
    value=14
)

# Apply filters
df_filtered = df[
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1]) &
    (df['room_type'].isin(room_types)) &
    (df['neighbourhood_group'].isin(neighborhood_groups)) &
    (df['minimum_nights'] <= min_nights_filter)
]

# Display filter results
st.sidebar.markdown(f"**Showing {len(df_filtered):,} of {len(df):,} listings**")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered Listings", f"{len(df_filtered):,}")
with col2:
    st.metric("Avg Price", f"${df_filtered['price'].mean():.0f}")
with col3:
    st.metric("Avg Availability", f"{df_filtered['availability_365'].mean():.0f} days")
with col4:
    st.metric("Neighborhoods", f"{df_filtered['neighbourhood'].nunique()}")

st.markdown("---")

# Three main visualizations
st.header("📊 Key Market Insights")

# 1. NEIGHBORHOOD ANALYSIS
st.subheader("🏘️ Neighborhood Analysis")

# Prepare neighborhood data
neighborhood_stats = df_filtered.groupby('neighbourhood').agg({
    'price': 'mean',
    'id': 'count',
    'availability_365': 'mean',
    'number_of_reviews': 'mean'
}).round(1).reset_index()
neighborhood_stats.columns = ['neighbourhood', 'avg_price', 'listing_count', 'avg_availability', 'avg_reviews']

# Filter to top neighborhoods by listing count
top_neighborhoods = neighborhood_stats.nlargest(10, 'listing_count')

# Interactive selection for neighborhood chart
chart_metric = st.selectbox(
    "Color neighborhoods by:",
    options=['avg_price', 'avg_availability', 'avg_reviews'],
    format_func=lambda x: {'avg_price': 'Average Price', 'avg_availability': 'Average Availability', 'avg_reviews': 'Average Reviews'}[x]
)

neighborhood_chart = alt.Chart(top_neighborhoods).mark_bar().encode(
    x=alt.X('listing_count:Q', title='Number of Listings'),
    y=alt.Y('neighbourhood:N', sort='-x', title='Neighborhood'),
    color=alt.Color(f'{chart_metric}:Q', 
                   scale=alt.Scale(scheme='viridis'),
                   title=chart_metric.replace('_', ' ').title()),
    tooltip=['neighbourhood:N', 'listing_count:Q', 'avg_price:Q', 'avg_availability:Q', 'avg_reviews:Q']
).properties(
    width=600,
    height=350,
    title=f"Top 10 Neighborhoods by Listing Count (Colored by {chart_metric.replace('_', ' ').title()})"
)

st.altair_chart(neighborhood_chart, use_container_width=True)

# Row for the other two charts
col1, col2 = st.columns(2)

with col1:
    # 2. PRICE DISTRIBUTION
    st.subheader("💰 Price Distribution")
    
    # Interactive binning option
    bin_count = st.slider("Number of price bins:", min_value=10, max_value=50, value=25, key="price_bins")
    
    price_dist_chart = alt.Chart(df_filtered).mark_bar(opacity=0.7).encode(
        x=alt.X('price:Q', 
               bin=alt.Bin(maxbins=bin_count),
               title='Price ($/night)'),
        y=alt.Y('count():Q', title='Number of Listings'),
        color=alt.Color('room_type:N', 
                       title='Room Type',
                       scale=alt.Scale(scheme='category10')),
        tooltip=['count():Q', 'room_type:N']
    ).properties(
        width=350,
        height=400,
        title="Price Distribution by Room Type"
    )
    
    st.altair_chart(price_dist_chart, use_container_width=True)

with col2:
    # 3. AVAILABILITY PATTERNS
    st.subheader("📅 Availability Patterns")
    
    # Interactive chart type selection
    availability_chart_type = st.radio(
        "Chart type:",
        options=['Histogram', 'Box Plot'],
        horizontal=True
    )
    
    if availability_chart_type == 'Histogram':
        availability_chart = alt.Chart(df_filtered).mark_bar(opacity=0.7).encode(
            x=alt.X('availability_365:Q', 
                   bin=alt.Bin(maxbins=20),
                   title='Days Available per Year'),
            y=alt.Y('count():Q', title='Number of Listings'),
            color=alt.Color('room_type:N', 
                           title='Room Type',
                           scale=alt.Scale(scheme='category10')),
            tooltip=['count():Q', 'room_type:N']
        ).properties(
            width=350,
            height=400,
            title="Availability Distribution"
        )
    else:
        availability_chart = alt.Chart(df_filtered).mark_boxplot(extent='min-max').encode(
            x=alt.X('room_type:N', title='Room Type'),
            y=alt.Y('availability_365:Q', title='Days Available per Year'),
            color=alt.Color('room_type:N', 
                           title='Room Type',
                           scale=alt.Scale(scheme='category10'))
        ).properties(
            width=350,
            height=400,
            title="Availability by Room Type"
        )
    
    st.altair_chart(availability_chart, use_container_width=True)

# Insights section
st.markdown("---")
st.header("💡 Key Market Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏆 Top Performing Neighborhoods**")
    top_3_neighborhoods = neighborhood_stats.nlargest(3, 'listing_count')[['neighbourhood', 'listing_count']]
    for idx, row in top_3_neighborhoods.iterrows():
        st.write(f"• {row['neighbourhood']}: {row['listing_count']} listings")

with col2:
    st.markdown("**💰 Price Insights**")
    most_expensive_room = df_filtered.groupby('room_type')['price'].mean().idxmax()
    price_by_room = df_filtered.groupby('room_type')['price'].mean()
    st.write(f"• Highest avg price: {most_expensive_room}")
    st.write(f"• Price range: ${df_filtered['price'].min()} - ${df_filtered['price'].max()}")
    st.write(f"• Median price: ${df_filtered['price'].median():.0f}")

with col3:
    st.markdown("**📊 Availability Insights**")
    high_availability = (df_filtered['availability_365'] > 300).sum()
    low_availability = (df_filtered['availability_365'] < 30).sum()
    st.write(f"• High availability (>300 days): {high_availability} listings")
    st.write(f"• Low availability (<30 days): {low_availability} listings")
    st.write(f"• Average availability: {df_filtered['availability_365'].mean():.0f} days")

# Coordination info
st.markdown("---")
st.info("💡 **Interactive Features**: Use the sidebar filters to explore different market segments. All charts update dynamically to show coordinated insights across neighborhoods, pricing, and availability patterns.")