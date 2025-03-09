

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sns.set(style='dark')

# Load dataset
@st.cache_data
def load_data():
    customers_df = pd.read_csv("analisis_ecommerce.csv")  # Sesuaikan dengan file dataset kamu
    geolocation_df = pd.read_csv("analisis_ecommerce.csv")
    products_df = pd.read_csv("analisis_ecommerce.csv")
    order_items_df = pd.read_csv("analisis_ecommerce.csv")
    orders_df = pd.read_csv("analisis_ecommerce.csv")
    return customers_df, geolocation_df, products_df, order_items_df, orders_df

customers_df, geolocation_df, products_df, order_items_df, orders_df = load_data()

# Distribusi pelanggan berdasarkan state
st.title("Dashboard Analisis Data E-commerce")

st.subheader("Distribusi Pelanggan Berdasarkan State")
state_counts = customers_df["customer_state"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=state_counts.index, y=state_counts.values, palette="viridis", ax=ax)
ax.set_xlabel("State")
ax.set_ylabel("Jumlah Pelanggan")
st.pyplot(fig)

# 10 kota dengan pelanggan terbanyak
st.subheader("10 Kota dengan Pelanggan Terbanyak")
top_cities = customers_df["customer_city"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_cities.index, y=top_cities.values, palette="coolwarm", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel("City")
ax.set_ylabel("Jumlah Pelanggan")
st.pyplot(fig)

# 7 kategori produk paling laku
st.subheader("7 Kategori Produk Paling Laku")
top_categories = products_df["product_category_name"].value_counts().head(7)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(y=top_categories.index, x=top_categories.values, palette="Blues_r", ax=ax)
ax.set_xlabel("Jumlah Penjualan")
ax.set_ylabel("Kategori Produk")
st.pyplot(fig)

# 7 kategori produk paling tidak laku
st.subheader("7 Kategori Produk Paling Tidak Laku")
least_categories = products_df["product_category_name"].value_counts().tail(7)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(y=least_categories.index, x=least_categories.values, palette="Reds_r", ax=ax)
ax.set_xlabel("Jumlah Penjualan")
ax.set_ylabel("Kategori Produk")
st.pyplot(fig)


#menambahkan header pada dashboard
st.header('Analisis Lanjutan "RFM Analysis" :sparkles:')



# memastikan kolom yang digunakan tersedia dalam dataset
orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"])

# mengitung nilai RFM
max_date = orders_df["order_purchase_timestamp"].max()

rfm = orders_df.groupby("customer_id").agg({
    "order_purchase_timestamp": lambda x: (max_date - x.max()).days, 
    "order_id": "count",
    "price": "sum"
}).reset_index()

rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]

# Sampling untuk plot
sample_size = 5000
rfm_sample = rfm.sample(n=sample_size, random_state=42) if len(rfm) > sample_size else rfm

# Dashboard di Streamlit
st.title("Dashboard RFM Analysis")

# Distribusi Recency
st.subheader("Distribusi Recency (Hari)")
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(rfm["Recency"], bins=50, kde=True, color="blue", ax=ax)
st.pyplot(fig)

# Distribusi Frequency
st.subheader("Distribusi Frequency (Jumlah Order)")
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(rfm["Frequency"], bins=50, kde=True, color="green", ax=ax)
st.pyplot(fig)

# Distribusi Monetary
st.subheader("Distribusi Monetary (Total Spending)")
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(rfm["Monetary"], bins=50, kde=True, color="red", ax=ax)
st.pyplot(fig)

# Boxplot untuk melihat outlier
st.subheader("Boxplot RFM Metrics")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.boxplot(y=rfm["Recency"], color="blue", ax=axes[0])
axes[0].set_title("Recency")

sns.boxplot(y=rfm["Frequency"], color="green", ax=axes[1])
axes[1].set_title("Frequency")

sns.boxplot(y=rfm["Monetary"], color="red", ax=axes[2])
axes[2].set_title("Monetary")

st.pyplot(fig)

# Scatter Plot Recency vs Frequency
st.subheader("Scatter Plot: Recency vs Frequency")
fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(x=rfm["Recency"], y=rfm["Frequency"], color="orange", ax=ax)
st.pyplot(fig)
