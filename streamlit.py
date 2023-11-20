import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("predictive_maintenance.csv")

# Streamlit app
st.title("Predictive Maintenance Data Analysis")

# Sidebar for user interaction, if needed
# Example: Select a feature for analysis
selected_feature = st.sidebar.selectbox("Select a feature for analysis", df.columns)

# Main content
st.subheader("Histogram with KDE Plot")

# Create a histogram with KDE plot using Seaborn
fig, ax = plt.subplots()
sns.histplot(data=df, x=selected_feature, hue="Target", kde=True, ax=ax)
st.pyplot(fig)

# Display data table
st.subheader("Data Table")
st.dataframe(df)

# You can add more sections or interactive elements based on your needs
# For example, you can add sliders, checkboxes, etc. to allow users to interact with the data

