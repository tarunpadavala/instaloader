from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import xgboost
import instaloader
import pandas as pd
import os

# Load the trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    username = request.form.get('IQ')
    
    # Define where to save/load the session file (e.g., use a temporary directory on Render)
    session_file = "session-tarun_paspuleti"
    
    # You can adjust the path if needed. For example, saving to /tmp on Render:
    temp_path = os.path.join('/tmp/.instaloader-render', session_file)
    
    L = instaloader.Instaloader()

    # Try loading session from the specific path
    try:
        L.load_session_from_file(temp_path)
    except FileNotFoundError:
        # If session file not found, you can log in and save a new session
        username1 = "tarun_paspuleti"
        password = "shivayanama17"
        
        L.login(username1, password)
        L.save_session_to_file(temp_path)

    try:
        profile = instaloader.Profile.from_username(L.context, username.strip().lower())
    except instaloader.exceptions.ProfileNotExistsException:
        return jsonify({"error": "The provided profile does not exist!"}), 404
    except instaloader.exceptions.ConnectionException:
        return jsonify({"error": "Profile Doesn't Exist"}), 503
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    profile_info = {
        "profile_pic": check(profile.profile_pic_url),
        "username_length": len(profile.username),
        "fullname_words": len(profile.full_name.split()),
        "fullname_length": len(profile.full_name),
        "name_equals_username": nameOk(profile.username, profile.full_name),
        "bio_length": len(profile.biography),
        "external_url": Pk(bool(profile.external_url)),
        "is_private": Pk(profile.is_private),
        "num_posts": profile.mediacount,
        "num_followers": profile.followers,
        "num_follows": profile.followees,
    }

    profile_df = pd.DataFrame([profile_info])
    prediction = model.predict(profile_df)[0]
    output = 'Not Fake' if prediction[0] == 1 else 'Fake'

    return render_template('index.html', prediction_text=f'Prediction: {output}')


if __name__ == "__main__":
    app.run(debug=True)
