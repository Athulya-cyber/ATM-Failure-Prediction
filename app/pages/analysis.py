import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

st.title("📊 Exploratory Data Analysis")

st.write("""Exploratory Data Analysis (EDA) is performed to understand the ATM dataset
before building the machine learning model. It helps identify the structure
and characteristics of the data, detect missing values and outliers, analyze
the distribution of variables, and understand relationships between features
and ATM failures."""
"""EDA also helps in selecting appropriate preprocessing techniques and
identifying important patterns that may contribute to ATM failure prediction.
""")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ATM_Maintenance_Cleaned.csv"

atm = pd.read_csv(DATA_PATH)

show_data = st.checkbox("Show Dataset")
if show_data:
    st.dataframe(atm)


st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Rows",
        atm.shape[0])

with col2:
    st.metric(
        "Columns",
        atm.shape[1])

st.dataframe(
    atm.head(),
    use_container_width=True)

st.markdown("---")


#---------------------------------------
# EDA 
#---------------------------------------
st.subheader("Outlier Analysis")
for col in atm.select_dtypes(include="number").columns:
    fig = plt.figure(figsize=(6,3))
    sns.boxplot(x=atm[col])
    plt.title(col)
    st.pyplot(fig)
    plt.close(fig)
    




st.subheader("Correlation Heatmap")
fig = plt.figure(figsize=(16,12))
sns.heatmap(atm.corr(numeric_only=True),
            cmap="coolwarm",
            annot=True)

st.pyplot(fig)
plt.close(fig)





st.subheader("Count Plot of Categorical Features")
fig = plt.figure(figsize=(18,12))

plt.subplot(2,3,1)
sns.countplot(data=atm,x="Manufacturer")
plt.xticks(rotation=90)

plt.subplot(2,3,2)
sns.countplot(data=atm,x="District")
plt.xticks(rotation=90)

plt.subplot(2,3,3)
sns.countplot(data=atm,x="Location_Type")
plt.xticks(rotation=90)

plt.subplot(2,3,4)
sns.countplot(data=atm,x="Failure_Next_48H")
plt.xticks(rotation=90)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)





st.subheader("Relationship Between Categorical Features")
fig = plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
sns.countplot(data=atm, x="Manufacturer", hue="Failure_Next_48H")
plt.title("Manufacturer vs Failure in Next 48 Hours")
plt.xticks(rotation=45)

plt.subplot(1,3,2)
sns.countplot(data=atm, x="District", hue="Failure_Next_48H")
plt.title("District vs Failure in Next 48 Hours")
plt.xticks(rotation=90)

plt.subplot(1,3,3)
sns.countplot(data=atm, x="Location_Type", hue="Failure_Next_48H")
plt.title("Location_Typevs Failure in Next 48 Hours")
plt.xticks(rotation=45)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)




st.subheader("Relationship between Failure_Next_48H and Numerical features")
fig = plt.figure(figsize=(10,5))

plt.subplot(2,3,1)
sns.boxplot(data=atm, x="Failure_Next_48H", y="CPU_Temp_C")

plt.subplot(2,3,2)
sns.boxplot(data=atm, x="Failure_Next_48H", y="Cash_Remaining_Pct")

plt.subplot(2,3,3)
sns.boxplot(data=atm, x="Failure_Next_48H", y="Daily_Transactions")

plt.subplot(2,3,4)
sns.boxplot(data=atm, x="Failure_Next_48H", y="Days_Since_Maintenance")

plt.subplot(2,3,5)
sns.boxplot(data=atm, x="Failure_Next_48H", y="Network_Drops")

plt.subplot(2,3,6)
sns.boxplot(data=atm, x="Failure_Next_48H", y="Customer_Complaints_7D")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)




st.subheader("Scatter Plot")
fig = plt.figure(figsize=(10,5))
sns.scatterplot(x="CPU_Temp_C", y="Daily_Transactions", hue="Failure_Next_48H", data=atm)
st.pyplot(fig)
plt.close(fig)
