import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("predictive_maintenance.csv")

# Streamlit app
st.title("Predictive Maintenance Data Analysis")

# Sidebar for user interaction, if needed
selected_feature = st.sidebar.selectbox("Select a feature for analysis", df.columns)

# Main content
st.subheader("Histogram with KDE Plot")

# Extract the selected column as a Pandas Series
selected_series = df[selected_feature]

# Convert the series to a NumPy array
selected_array = selected_series.to_numpy()

# Create a histogram with KDE plot using Seaborn
fig, ax = plt.subplots()
sns.histplot(data=selected_array, kde=True, ax=ax)
st.pyplot(fig)

# Display data table
st.subheader("Data Table")
st.dataframe(df)

# You can add more sections or interactive elements based on your needs
