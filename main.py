from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, uuid, zipfile, shutil, io, pickle
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import joblib
from pymongo import MongoClient
import pandas as pd
import datetime

app = FastAPI()

# Connect to MongoDB (adjust URI as needed)
mongo_client = MongoClient("mongodb+srv://fakinwande:M50xQRyrwpnBGG9j@cluster0.f8wa06y.mongodb.net/")
db = mongo_client["environmental_sounds"]
retrain_collection = db["retrain_data"]
models_collection = db["models"]

# Function to load latest model from MongoDB
def load_latest_model():
    try:
        # Get the most recent model from MongoDB
        latest_model = models_collection.find_one(
            sort=[("timestamp", -1)]
        )
        
        if latest_model:
            # Load model from bytes
            model_buffer = io.BytesIO(latest_model["model_data"])
            model = load_model(model_buffer)
            
            # Load encoders from bytes
            le = pickle.loads(latest_model["label_encoder"])
            scaler = pickle.loads(latest_model["scaler"])
            
            print(f"Loaded model version: {latest_model['version']}")
            print("Label classes:", le.classes_)
            
            return model, le, scaler
        else:
            # Fallback to filesystem if no model in database
            print("No model found in database, loading from filesystem...")
            model = load_model("audio_classifier_model.h5")
            le = joblib.load("label_encoder.pkl")
            scaler = joblib.load("scaler.pkl")
            print("Label classes:", le.classes_)
            return model, le, scaler
    except Exception as e:
        print(f"Error loading from database: {e}")
        print("Falling back to filesystem...")
        model = load_model("audio_classifier_model.h5")
        le = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        print("Label classes:", le.classes_)
        return model, le, scaler

# Load existing model & encoders
model, le, scaler = load_latest_model()

RETRAIN_DIR = "retrain_data"

# Timestamp for versioning models
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Feature Extraction ----------
def extract_features(file_path, n_mfcc=13, n_chroma=12, n_mel=128):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y)
        if len(y) == 0:
            return None
        y = y / (np.max(np.abs(y)) + 1e-6)

        features = []

        # Basic MFCC features (compatible with original model)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features.extend([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.max(mfcc, axis=1), np.min(mfcc, axis=1)])

        # Basic chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        features.extend([np.mean(chroma, axis=1), np.std(chroma, axis=1)])

        # Basic mel spectrogram features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])

        # Basic spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        features.extend([
            np.mean(spectral_centroids), np.std(spectral_centroids),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
        ])

        return np.concatenate([np.array(f).flatten() for f in features])
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# ---------- Predict Endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    features = extract_features(temp_filename)
    os.remove(temp_filename)

    if features is None:
        return JSONResponse(status_code=400, content={"error": "Invalid audio file."})

    scaled = scaler.transform([features])
    probs = model.predict(scaled)
    pred_index = np.argmax(probs, axis=1)[0]
    pred_label = le.inverse_transform([pred_index])[0]

    return {"prediction": pred_label}

# ---------- Retrain Endpoint (Upload ZIP + Retrain) ----------
@app.post("/retrain")
async def retrain_model(zipfile_data: UploadFile = File(...)):
    # Read the uploaded zip file
    zip_content = await zipfile_data.read()
    
    # Parse zip file in memory
    with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
        # Find CSV file in the zip
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            return {"error": "No CSV file found in the uploaded data."}
        
        # Read CSV from zip
        csv_content = zip_ref.read(csv_files[0])
        df = pd.read_csv(io.BytesIO(csv_content))
        
        # Determine which column contains the labels
        possible_label_columns = ['label', 'category', 'class', 'target']
        label_column = None
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            return {"error": f"No label column found. Available columns: {list(df.columns)}"}
        
        features, labels = [], []
        
        # Process audio files from zip
        for _, row in df.iterrows():
            file = row['filename']
            label = row[label_column]
            
            # Check if audio file exists in zip
            if file in zip_ref.namelist() and file.endswith(".wav"):
                # Read audio file from zip
                audio_content = zip_ref.read(file)
                
                # Save temporarily for librosa processing
                temp_audio_path = f"temp_{uuid.uuid4().hex}.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_content)
                
                fvec = extract_features(temp_audio_path)
                
                # Clean up temp file with proper error handling
                try:
                    os.remove(temp_audio_path)
                except PermissionError:
                    # If file is still in use, try again after a short delay
                    import time
                    time.sleep(0.1)
                    try:
                        os.remove(temp_audio_path)
                    except:
                        pass  # Ignore if still can't delete
                except:
                    pass  # Ignore other deletion errors
                
                if fvec is not None:
                    features.append(fvec)
                    labels.append(label)
                    
                    # Store in MongoDB with audio data
                    retrain_collection.insert_one({
                        "filename": file,
                        "label": label,
                        "features": fvec.tolist(),
                        "audio_data": audio_content,  # Store the actual audio file
                        "upload_timestamp": datetime.datetime.now(),
                        "metadata": row.to_dict()  # Store all CSV row data
                    })

    if len(features) < 2:
        return {"error": "Not enough valid data to retrain."}

    X = np.array(features)
    y = np.array(labels)

    le_new = LabelEncoder()
    y_encoded = le_new.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    scaler_new = StandardScaler()
    X_scaled = scaler_new.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size=0.2, stratify=y_encoded)

    # Load existing model for transfer learning
    base_model = load_model("audio_classifier_model.h5")
    
    # Use the base model as starting point
    model_new = Sequential()
    
    # Copy all layers from base model except the output layer
    for layer in base_model.layers[:-1]:
        model_new.add(layer)
    
    # Add the new output layer with correct number of classes
    model_new.add(Dense(len(le_new.classes_), activation='softmax'))
    
    # Optionally freeze some layers for transfer learning
    # Unfreeze the last few layers for fine-tuning
    for layer in model_new.layers[-3:]:  # Unfreeze last 3 layers
        layer.trainable = True
    for layer in model_new.layers[:-3]:  # Freeze earlier layers
        layer.trainable = False

    # Use better optimizer with learning rate scheduling
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    
    optimizer = Adam(learning_rate=0.001)
    model_new.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Add callbacks for better training
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    # Train with more epochs and better batch size
    history = model_new.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=30, 
        batch_size=32,
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    # Get final accuracy values
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]

    # Evaluate on all data for overall accuracy and loss
    overall_loss, overall_accuracy = model_new.evaluate(X_scaled, y_cat, verbose=0)

    # Save model and encoders to MongoDB
    # Save model as HDF5 bytes using temporary file
    temp_model_path = f"temp_model_{uuid.uuid4().hex}.h5"
    model_new.save(temp_model_path)
    
    # Read the saved model file
    with open(temp_model_path, 'rb') as f:
        model_bytes = f.read()
    
    # Clean up temporary file
    os.remove(temp_model_path)
    
    # Serialize encoders using pickle
    le_bytes = pickle.dumps(le_new)
    scaler_bytes = pickle.dumps(scaler_new)
    
    # Store in MongoDB with metadata
    model_doc = {
        "timestamp": datetime.datetime.now(),
        "model_data": model_bytes,
        "label_encoder": le_bytes,
        "scaler": scaler_bytes,
        "accuracy": float(overall_accuracy),
        "loss": float(overall_loss),
        "classes": le_new.classes_.tolist(),
        "version": timestamp
    }
    
    # Insert new model version
    models_collection.insert_one(model_doc)

    # Clean up any remaining temporary files
    import time
    for temp_file in os.listdir("."):
        if temp_file.startswith("temp_") and (temp_file.endswith(".wav") or temp_file.endswith(".h5")):
            try:
                os.remove(temp_file)
            except PermissionError:
                # Wait a bit and try again
                time.sleep(0.1)
                try:
                    os.remove(temp_file)
                except:
                    pass  # Give up if still can't delete
            except:
                pass  # Ignore other errors

    return {
        "message": "Model retrained and updated successfully.",
        "overall_accuracy": float(overall_accuracy),
        "overall_loss": float(overall_loss)
    }

@app.get("/")
async def root():
    return {"message": "Environmental Sounds API is running."}
