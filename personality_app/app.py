# app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Set style for seaborn
sns.set(style="whitegrid")

# Set page config
st.set_page_config(page_title="Personality Predictor", layout="wide")

# Load model and scaler
model = pickle.load(open("personality_app/ensemble_model.pkl", "rb"))
scaler = pickle.load(open("personality_app/ensemble_scaler.pkl", "rb"))

# Define personality map
personality_map = {0: "Ambivert", 1: "Extrovert", 2: "Introvert"}

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["🧠 Personality Predictor", "📊 Dashboard", '🔍 Clustering'])

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

    if st.button("Predict Personality"):
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        st.subheader("Predicted Personality Type:")
        st.success(personality_map[prediction])

# -------- TAB 2: Dashboard --------
elif selection == "📊 Dashboard":
    st.title("📊 Dataset Dashboard & Insights")

    uploaded_file = st.file_uploader("Upload the original dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📁 Data Preview")
        st.dataframe(df.head())

        # Target distribution
        st.subheader("🧮 Personality Type Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='personality_type', data=df, ax=ax1, palette='Set2')
        st.pyplot(fig1)

        # Correlation heatmap
        st.subheader("📌 Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.drop('personality_type', axis=1).corr(), cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

        # Boxplot
        st.subheader("📦 Boxplot of Features")
        selected_col = st.selectbox("Select Feature", df.select_dtypes(include='float').columns)
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='personality_type', y=selected_col, data=df, ax=ax3, palette='Set3')
        st.pyplot(fig3)

# -------- TAB 3: Clustering --------
elif selection == "🔍 Clustering":
    st.title("🔍 Clustering Behavioral Traits")

    uploaded_file = st.file_uploader("Upload the original dataset (CSV)", type=["csv"], key="clustering")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded!")
        
        if 'personality_type' in df.columns:
            features = df.drop(['personality_type'], axis=1)
        else:
            features = df

        # Scaling
        scaler_clustering = StandardScaler()
        X_scaled = scaler_clustering.fit_transform(features)

        # PCA for plotting
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Clustering options
        st.subheader("Clustering Settings")
        algo = st.selectbox("Choose clustering algorithm", ["KMeans"])  # Add more later

        if algo == "KMeans":
            k = st.slider("Number of Clusters (k)", 2, 10, 3)
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)

            # Plot clusters
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=60)
            plt.title(f"{algo} Clustering Results (PCA)")
            st.pyplot(fig)

            # Display cluster counts
            st.subheader("📊 Cluster Distribution")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            st.write(cluster_counts)

            # Silhouette score
            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X_scaled, labels)
                st.info(f"Silhouette Score: {score:.3f}")
            else:
                st.warning("Cannot calculate silhouette score (only one cluster or noise).")

            clustered_df = df.copy()
            clustered_df['cluster'] = labels

            csv = clustered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")

