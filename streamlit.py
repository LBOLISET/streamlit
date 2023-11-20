import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Create a histogram with KDE plot using Matplotlib
fig, ax = plt.subplots()
ax.hist(selected_array, bins='auto', density=True, alpha=0.7, color='blue', label='Histogram')
kde = st.gaussian_kde(selected_array)
x_vals = np.linspace(selected_array.min(), selected_array.max(), 100)
ax.plot(x_vals, kde(x_vals), color='red', label='KDE')
ax.set_xlabel(selected_feature)
ax.legend()

# Display the plot
st.pyplot(fig)

# Display data table
st.subheader("Data Table")
st.dataframe(df)

# You can add more sections or interactive elements based on your needs
