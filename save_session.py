import instaloader

L = instaloader.Instaloader()
username1 = "tarun_paspuleti"
password = "shivayanama17"

L.login(username1, password)  # Logs in
L.save_session_to_file()  # Save session for future use
print("Session saved successfully!")
