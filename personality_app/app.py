import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

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

    if st.button("Predict Personality Cluster"):
        input_scaled = scaler.transform([user_input])
        cluster = model.predict(input_scaled)[0]
        label, insight = personality_map.get(cluster, (f"Cluster {cluster}", "No description available."))
        st.subheader("Predicted Personality Cluster:")
        st.success(f"{label}")
        st.info(insight)

        st.subheader("Your Personality Radar Chart")
        radar_labels = [f.replace('_', ' ').title() for f in feature_labels[:8]]
        radar_values = user_input[:8]
        radar_chart = f"""
        <canvas id="radarChart" width="350" height="350"></canvas>
        <script>
        const ctx = document.getElementById('radarChart').getContext('2d');
        new Chart(ctx, {{
            type: 'radar',
            data: {{
                labels: {radar_labels},
                datasets: [{{
                    label: 'Your Profile',
                    data: {radar_values},
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{responsive: true, maintainAspectRatio: false}}
        }});
        </script>
        """
        components.html(radar_chart, height=400)

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

        X_scaled = scaler.transform(df)
        labels = model.predict(X_scaled)
        df['cluster'] = labels

        st.subheader("Cluster Distribution")
        cluster_counts = df['cluster'].value_counts().sort_index().tolist()
        cluster_labels = [f"Cluster {i}" for i in range(len(cluster_counts))]
        bar_chart = f"""
        <canvas id='barChart'></canvas>
        <script>
        new Chart(document.getElementById('barChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: {cluster_labels},
                datasets: [{{
                    label: 'Count',
                    data: {cluster_counts},
                    backgroundColor: ['#66c2a5','#fc8d62','#8da0cb']
                }}]
            }},
            options: {{responsive: true, maintainAspectRatio: false}}
        }});
        </script>
        """
        components.html(bar_chart, height=350)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]

        st.subheader("PCA-based Cluster Visualization")
        scatter_data = df[['pca1', 'pca2', 'cluster']].values.tolist()
        scatter_chart = f"""
        <canvas id='scatterChart'></canvas>
        <script>
        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c'];
        const grouped = {{}};
        {scatter_data}.forEach(function(point) {{
            const [x, y, c] = point;
            if (!grouped[c]) grouped[c] = [];
            grouped[c].push({{x: x, y: y}});
        }});
        const datasets = Object.entries(grouped).map(function([key, data], idx) {{
            return {{
                label: 'Cluster ' + key,
                data: data,
                backgroundColor: colors[idx % colors.length]
            }};
        }});
        new Chart(document.getElementById('scatterChart').getContext('2d'), {{
            type: 'scatter',
            data: {{ datasets: datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'PCA 1' }} }},
                    y: {{ title: {{ display: true, text: 'PCA 2' }} }}
                }}
            }}
        }});
        </script>
        """

        components.html(scatter_chart, height=400)

        st.subheader("Cluster-wise Feature Means (Line Plot)")
        cluster_means = df.groupby('cluster').mean()
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))

        st.subheader("Violin Plot: Feature Distribution by Cluster")
        feat_col = st.selectbox("Select Feature", df.select_dtypes(include='number').columns.drop(['cluster', 'pca1', 'pca2']))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.violinplot(x='cluster', y=feat_col, data=df, palette='Set3', ax=ax)
        st.pyplot(fig)

        st.subheader("Cluster Descriptions")
        for cid, (name, desc) in personality_map.items():
            st.markdown(f"**Cluster {cid} - {name}:** {desc}")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Dataset", csv, "clustered_personality_data.csv", "text/csv")

    except FileNotFoundError:
        st.error("The required dataset 'personality_app/personality.csv' was not found. Please make sure it exists.")
