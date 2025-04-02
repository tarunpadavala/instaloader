from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import xgboost
import instaloader
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    username  = request.form.get('IQ')  # Fetch single value

    # Load session instead of logging in every time
    L = instaloader.Instaloader()
    L.load_session_from_file("tarun_paspuleti")

    try:
        profile = instaloader.Profile.from_username(L.context, username.strip().lower())
    except instaloader.exceptions.ProfileNotExistsException:
        return jsonify({"error": "The provided profile does not exist!"}), 404
    except instaloader.exceptions.ConnectionException:
        return jsonify({"error": "Profile Doesn't Exist"}), 503
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    profile_info = {
        "username_length": len(profile.username),
        "fullname_words": len(profile.full_name.split()),
        "fullname_length": len(profile.full_name),
        "bio_length": len(profile.biography),
        "num_posts": profile.mediacount,
        "num_followers": profile.followers,
        "num_follows": profile.followees,
    }

    profile_df = pd.DataFrame([profile_info])
    prediction = model.predict(profile_df)[0]

    output = 'Not Fake' if prediction == 1 else 'Fake'
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
