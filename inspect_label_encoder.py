import joblib

encoder_path = "label_encoder.pkl"  # Change path if needed

le = joblib.load(encoder_path)
print("LabelEncoder classes:", le.classes_)