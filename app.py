from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import xgboost
import instaloader
import pandas as pd
# Load the trained model


# Load the model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    int_feature = request.form.get('IQ')  # Fetch single value directly
    username = np.array([int_feature])  # Convert it into a NumPy array
    L = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except instaloader.exceptions.ProfileNotExistsException:
        return jsonify(({"error": "The provided profile does not exist!"}, status=404)
    except instaloader.exceptions.ConnectionException:
        return jsonify{"error": "Profile Doesnt Exists"}, status=503)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}, status=500)
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
                # "is_verified": Pk(profile.is_verified),
            }
            
    print("Bio: ",profile.biography)
            
            
                
    profile_df = pd.DataFrame([profile_info])
    prediction = model.predict(profile_df)[0]
            # print("Prediction",prediction)
      
# Make prediction
   # prediction = model.predict(final_features)
    output = 'Not Fake' if prediction[0] == 1 else 'Fake'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
