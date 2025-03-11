import joblib
import requests

# Download the model from Hugging Face
url = "https://huggingface.co/khfong26/nba_all_star_model/resolve/main/tuned_random_forest.pkl"
response = requests.get(url)

# Save it temporarily
with open("old_model.pkl", "wb") as f:
    f.write(response.content)

# Load the old model
old_model = joblib.load("old_model.pkl")

# Re-save it using Python 3.11 standards
joblib.dump(old_model, "tuned_random_forest.pkl")

print("Model successfully re-saved in Python 3.11 format!")
