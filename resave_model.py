import joblib
import requests
from sklearn.ensemble import RandomForestClassifier  # or whatever model you used

# Download the model from Hugging Face
url = "https://huggingface.co/khfong26/nba_all_star_model/resolve/main/tuned_random_forest.pkl"
response = requests.get(url)

# Save it temporarily
with open("old_model.pkl", "wb") as f:
    f.write(response.content)

# Load the old model
old_model = joblib.load("old_model.pkl")

# Re-save the model without serialization issues
joblib.dump(old_model, "tuned_random_forest.pkl", compress=3)

print("Model re-saved successfully!")
