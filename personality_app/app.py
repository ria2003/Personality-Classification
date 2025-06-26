import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# App Configuration
st.set_page_config(page_title="Personality Predictor", layout="wide")
sns.set(style="whitegrid")

# Load models
kmeans_model = pickle.load(open("personality_app/kmeans_model.pkl", "rb"))
scaler = pickle.load(open("personality_app/ensemble_scaler.pkl", "rb"))

# Define personality labels and insights
personality_map = {0: "Ambivert", 1: "Extrovert", 2: "Introvert"}
personality_insights = {
    0: {
        "Description": "Ambiverts are a balanced mix of introverts and extroverts. They can enjoy social interactions, but also need alone time.",
        "Strengths": "Adaptability, flexibility, good communication.",
        "Challenges": "May feel conflicted in extreme social situations.",
        "Environment": "Thrive in balanced work/play environments."
    },
    1: {
        "Description": "Extroverts are outgoing, talkative, and enjoy social interactions. They recharge through engaging with others.",
        "Strengths": "High energy, enthusiasm, sociability.",
        "Challenges": "May struggle with solitude or introspection.",
        "Environment": "Collaborative, fast-paced, socially dynamic environments."
    },
    2: {
        "Description": "Introverts are thoughtful, reflective, and enjoy deep connections. They often need solitude to recharge.",
        "Strengths": "Focus, creativity, deep thinking.",
        "Challenges": "May struggle in overly social or noisy settings.",
        "Environment": "Quiet, individual-focused environments with autonomy."
    }
}

# Data persistence file
DATA_FILE = "personality_app/personality.csv"

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
selection = st.sidebar.radio("Go to", ["🧠 Personality Predictor", "📊 Dashboard"])

# Feature List
feature_labels = [
    "social_energy", "alone_time_preference", "talkativeness", "deep_reflection",
    "group_comfort", "party_liking", "listening_skill", "empathy", "creativity",
    "organization", "leadership", "risk_taking", "public_speaking_comfort",
    "curiosity", "routine_preference", "excitement_seeking", "friendliness",
    "emotional_stability", "planning", "spontaneity", "adventurousness",
    "reading_habit", "sports_interest", "online_social_usage", "travel_desire",
    "gadget_usage", "work_style_collaborative", "decision_speed", "stress_handling"
]

# TAB 1: Personality Prediction
if selection == "🧠 Personality Predictor":
    st.title("🧠 Live Personality Type Predictor")
    st.markdown("Adjust your behavioral traits to discover your personality type:")

    user_input = []
    cols = st.columns(3)
    for idx, feat in enumerate(feature_labels):
        with cols[idx % 3]:
            val = st.slider(f"{feat.replace('_', ' ').title()}", 0.0, 10.0, 5.0)
            user_input.append(val)

    # Centered Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Predict Personality"):
            input_df = pd.DataFrame([user_input], columns=feature_labels)
            scaled_input = scaler.transform(input_df)
            prediction = kmeans_model.predict(scaled_input)[0]
            label = personality_map[prediction]
            insight = personality_insights[prediction]

            st.subheader("🔮 Your Personality Type:")
            st.success(f"**{label}**")

            st.markdown(f"**Description:** {insight['Description']}")
            st.markdown(f"**Strengths:** {insight['Strengths']}")
            st.markdown(f"**Challenges:** {insight['Challenges']}")
            st.markdown(f"**Ideal Environment:** {insight['Environment']}")

            # Save input + prediction
            input_df = pd.DataFrame([user_input], columns=feature_labels)
            input_df['personality_type'] = label

            if os.path.exists(DATA_FILE):
                prev_data = pd.read_csv(DATA_FILE)
                updated = pd.concat([prev_data, input_df], ignore_index=True)
            else:
                updated = input_df

            updated.to_csv(DATA_FILE, index=False)

# TAB 2: Dashboard
elif selection == "📊 Dashboard":
    st.title("📊 Personality Dataset Dashboard")

    if not os.path.exists(DATA_FILE):
        st.warning("No data available yet. Use the Predictor tab first.")
    else:
        df = pd.read_csv(DATA_FILE)

        st.subheader("Recent Entries")
        st.dataframe(df.tail(10))

        st.subheader("Personality Type Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='personality_type', data=df, palette='Set2', ax=ax1)
        ax1.set_title("Count of Each Personality Type")
        st.pyplot(fig1)

        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.drop('personality_type', axis=1).corr(), cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Feature Comparison")
        selected_col = st.selectbox("Select a feature to compare:", df.select_dtypes(include='float').columns)
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='personality_type', y=selected_col, data=df, palette='Set3', ax=ax3)
        ax3.set_title(f"{selected_col} by Personality Type")
        st.pyplot(fig3)

        st.subheader("Top Traits per Type")
        top_traits = df.groupby("personality_type").mean().T
        top_sorted = top_traits.apply(lambda x: x.sort_values(ascending=False).head(5))
        st.dataframe(top_sorted)
