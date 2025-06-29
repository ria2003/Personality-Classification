import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

sns.set(style="whitegrid")

st.set_page_config(page_title="Personality Predictor", layout="wide")

model = pickle.load(open("personality_app/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("personality_app/ensemble_scaler.pkl", "rb"))

personality_map = {
    0: ("Introvert", "You tend to enjoy solitude, deep thinking, and quiet environments. You're thoughtful and often recharge by spending time alone."),
    1: ("Ambivert", "You display a balance of introverted and extroverted tendencies. You enjoy both social interactions and time alone."),
    2: ("Extrovert", "You thrive in social environments, love engaging with people, and gain energy from being around others.")
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["🧠 Personality Predictor", "📊 Dashboard"])

if not os.path.exists("user_inputs"):
    os.makedirs("user_inputs")

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
        cluster = model.predict(input_scaled)[0]
        label, insight = personality_map.get(cluster, (f"Cluster {cluster}", "No description available."))
        st.subheader("Predicted Personality :")
        st.success(f"{label}")
        st.info(insight)

        st.subheader("Your Personality Radar Chart")
        categories = [f.replace('_', ' ').title() for f in feature_labels[:29]]
        values = user_input[:29]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='purple', linewidth=2)
        ax.fill(angles, values, color='purple', alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=4)
        ax.set_yticklabels([])
        st.pyplot(fig)


        input_data = pd.DataFrame([user_input], columns=feature_labels)
        input_data['cluster'] = cluster
        input_data.to_csv("personality_app/personality.csv", mode='a', header=not os.path.exists("personality_app/personality.csv"), index=False)
        st.info("Your input has been saved for future model improvement.")
        


elif selection == "📊 Dashboard":
    st.title("📊 Dataset Dashboard & Insights")

    try:
        df = pd.read_csv("personality_app/personality.csv")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        if 'personality_type' in df.columns:
            df = df.drop('personality_type', axis=1)

        feature_labels = [
            "social_energy", "alone_time_preference", "talkativeness", "deep_reflection",
            "group_comfort", "party_liking", "listening_skill", "empathy", "creativity",
            "organization", "leadership", "risk_taking", "public_speaking_comfort",
            "curiosity", "routine_preference", "excitement_seeking", "friendliness",
            "emotional_stability", "planning", "spontaneity", "adventurousness",
            "reading_habit", "sports_interest", "online_social_usage", "travel_desire",
            "gadget_usage", "work_style_collaborative", "decision_speed", "stress_handling"
        ]

        X_scaled = scaler.transform(df[feature_labels])
        labels = model.predict(X_scaled)
        df['cluster'] = labels
        df['Personality'] = df['cluster'].map({k: v[0] for k, v in personality_map.items()})

        st.subheader("Cluster Distribution")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.countplot(x='Personality', data=df, palette='Set2', ax=ax1)
        st.pyplot(fig1)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]

        st.subheader("PCA-based Cluster Visualization")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.scatterplot(x='pca1', y='pca2', hue='Personality', data=df, palette='tab10', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Cluster-wise Feature Means")
        cluster_means = df.groupby('Personality')[feature_labels].mean()
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))

        st.subheader("Feature Violin Plot by Personality")
        feat_col = st.selectbox("Select Feature", df.select_dtypes(include='number').columns.drop(['cluster', 'pca1', 'pca2']))
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.violinplot(x='Personality', y=feat_col, data=df, palette='Set3', ax=ax3)
        st.pyplot(fig3)

        st.subheader("Top Features for Each Personality")
        top_n = 5
        top_features_df = pd.DataFrame()
        for persona, row in cluster_means.iterrows():
            top_feats = row.sort_values(ascending=False).head(top_n)
            temp_df = pd.DataFrame({"Feature": top_feats.index, "Score": top_feats.values, "Personality": persona})
            top_features_df = pd.concat([top_features_df, temp_df])

        fig4, ax4 = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Feature', y='Score', hue='Personality', data=top_features_df, ax=ax4)
        ax4.set_title("Top Features by Personality Type")
        st.pyplot(fig4)

        st.subheader("Cluster Descriptions")
        for cid, (name, desc) in personality_map.items():
            st.markdown(f"**{name}:** {desc}")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Dataset", csv, "clustered_personality_data.csv", "text/csv")

    except FileNotFoundError:
        st.error("The required dataset 'personality_app/personality.csv' was not found. Please make sure it exists.")
