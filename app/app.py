import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

import joblib
import requests

url = "https://huggingface.co/khfong26/nba_all_star_model/resolve/main/tuned_random_forest.pkl"

# Download the model when the app runs
model_path = "tuned_random_forest.pkl"
response = requests.get(url)
with open(model_path, "wb") as f:
    f.write(response.content)

# Load the model safely
model = joblib.load(model_path)


# Dynamically construct the absolute path to the dataset
data_path = os.path.join(os.path.dirname(__file__), '../data/merged_data.csv')
merged_df = pd.read_csv(data_path)

# Define the features used in the model
features = ["pts_reg", "asts_reg", "reb_reg", "stl_reg", "blk_reg", "fga_reg", "fta_reg", "tpa_reg"]

# Function to make a prediction
def predict_all_star(stats):
    df = pd.DataFrame([stats], columns=features)
    proba = model.predict_proba(df)[0][1]  # Probability of being an All-Star
    prediction = model.predict(df)[0]
    return prediction, proba

# Streamlit app
def main():
    st.title("NBA All-Star Prediction App")
    st.write("Input a player's regular season statistics to predict if they'll become an All-Star.")

    # Input fields (whole numbers)
    stats = []
    for feature in features:
        value = st.number_input(f"{feature.replace('_reg', '').upper()}", min_value=0, value=0, step=1)
        stats.append(value)

    # Predict button
    if st.button("Predict All-Star Status"):
        prediction, proba = predict_all_star(stats)
        if prediction == 1:
            st.success(f"✅ This player is predicted to be an All-Star with {proba:.2%} confidence.")
        else:
            st.error(f"❌ This player is predicted NOT to be an All-Star. Confidence: {proba:.2%}")

    # Random player prediction
    if st.button("Predict Random Player"):
        random_player = merged_df.sample(1)

        if "firstname_reg" in merged_df.columns and "lastname_reg" in merged_df.columns:
            player_name = f"{random_player['firstname_reg'].values[0]} {random_player['lastname_reg'].values[0]}"
        else:
            player_name = "Unknown Player"

        player_stats = random_player[features].iloc[0]
        prediction, proba = predict_all_star(player_stats)

        st.write(f"### Random Player: **{player_name}**")
        st.write(random_player[["firstname_reg", "lastname_reg"] + features])  # Show full name & stats

        if prediction == 1:
            st.success(f"✅ {player_name} is predicted to be an All-Star with {proba:.2%} confidence.")
        else:
            st.error(f"❌ {player_name} is predicted NOT to be an All-Star. Confidence: {proba:.2%}")


    # File upload for bulk predictions
    st.write("### Upload a CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if not set(features).issubset(data.columns):
            st.error("The uploaded CSV must contain the following columns:")
            st.write(features)
        else:
            predictions = model.predict(data[features])
            probabilities = model.predict_proba(data[features])[:, 1]
            data['All-Star Prediction'] = predictions
            data['All-Star Probability'] = probabilities
            st.write(data)
            st.download_button(
                label="Download Results as CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name='all_star_predictions.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
