import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv("predictive_maintenance.csv")

st.title("Predictive Maintenance Data Analysis")

selected_feature = st.sidebar.selectbox("Select a feature for analysis", df.columns)

st.subheader("Histogram with KDE Plot")

# Create a histogram with KDE plot using Seaborn
fig, ax = plt.subplots()
sns.histplot(data=df, x=selected_feature, hue="Target", kde=True, ax=ax)
st.pyplot(fig)

# Display data table
st.subheader("Data Table")
st.dataframe(df)
