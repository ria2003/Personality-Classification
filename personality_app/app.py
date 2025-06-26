import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Set style for seaborn
sns.set(style="whitegrid")

# Set page config
st.set_page_config(page_title="Personality Predictor", layout="wide")

# Load model and scaler
model = pickle.load(open("personality_app/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("personality_app/ensemble_scaler.pkl", "rb"))

# Define personality map
personality_map = {0: "Introvert", 1: "Extrovert", 2: "Ambivert"}

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["🧠 Personality Predictor", "📊 Dashboard"])

# Create a directory to store new inputs if it doesn't exist
if not os.path.exists("user_inputs"):
    os.makedirs("user_inputs")

# -------- TAB 1: Personality Predictor --------
if selection == "🧠 Personality Predictor":
    st.title("🧠 Live Personality Type Predictor")
    st.markdown("Select your behavioral traits below:")

    feature_labels = [
        "social_energy", "alone_time_preference", "talkativeness", "deep_reflection",
        "group_comfort", "party_liking", "listening_skill", "empathy", "creativity",
        "organization", "leadership", "risk_taking", "public_speaking_comfort",
        "curiosity", "routine_preference", "excitement_seeking", "friendliness",
        "emotional_stability", "planning", "spontaneity", "adventurousness",
        "reading_habit", "sports_interest", "online_social_usage", "travel_desire",
        "gadget_usage", "work_style_collaborative", "decision_speed", "stress_handling"
    ]

    user_input = []
    cols = st.columns(3)
    for idx, feat in enumerate(feature_labels):
        with cols[idx % 3]:
            val = st.slider(f"{feat.replace('_', ' ').title()}", 0.0, 10.0, 5.0)
            user_input.append(val)

    if st.button("Predict Personality Cluster"):
        input_scaled = scaler.transform([user_input])
        cluster = model[input_scaled][0] if isinstance(model, np.ndarray) else model.predict(input_scaled)[0]
        st.subheader("Predicted Personality Cluster:")
        st.success(f"You belong to {personality_map.get(cluster, f'Cluster {cluster}')}.")

        # Save the input for retraining
        input_data = pd.DataFrame([user_input], columns=feature_labels)
        input_data['cluster'] = cluster
        input_data.to_csv("user_inputs/new_input.csv", mode='a', header=not os.path.exists("user_inputs/new_input.csv"), index=False)
        st.info("Your input has been saved for future model improvement.")

# -------- TAB 2: Dashboard --------
elif selection == "📊 Dashboard":
    st.title("📊 Dataset Dashboard & Insights")

    try:
        df = pd.read_csv("personality_app/personality.csv")
        st.subheader("📁 Data Preview")
        st.dataframe(df.head())

        # Drop target if exists
        if 'personality_type' in df.columns:
            df = df.drop('personality_type', axis=1)

        X_scaled = scaler.transform(df)
        labels = model.predict(X_scaled)
        df['cluster'] = labels

        # Cluster distribution
        st.subheader("📊 Cluster Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='cluster', data=df, palette='Set2', ax=ax1)
        st.pyplot(fig1)

        # PCA for 2D scatter plot
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]

        st.subheader("🌀 PCA-based Cluster Visualization")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='tab10', ax=ax2)
        st.pyplot(fig2)

        # Cluster-wise means
        st.subheader("📈 Cluster-wise Feature Means")
        cluster_means = df.groupby('cluster').mean()
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))

        # Feature-wise boxplots by cluster
        st.subheader("📦 Boxplot for Selected Feature by Cluster")
        feat_col = st.selectbox("Select Feature", df.select_dtypes(include='number').columns.drop(['cluster', 'pca1', 'pca2']))
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='cluster', y=feat_col, data=df, palette='Set3', ax=ax3)
        st.pyplot(fig3)

        # Silhouette Score
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            st.info(f"Silhouette Score: {score:.3f}")

        # Download clustered dataset
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Dataset", csv, "clustered_personality_data.csv", "text/csv")

    except FileNotFoundError:
        st.error("The required dataset 'personality_app/personality_data.csv' was not found. Please make sure it exists.")
