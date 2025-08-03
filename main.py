from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os, zipfile, io, pickle, tempfile
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
from pymongo import MongoClient
import pandas as pd
import datetime

# Disable librosa caching to avoid permission issues in containers
os.environ['LIBROSA_CACHE_DIR'] = '/tmp'
os.environ['LIBROSA_USE_CACHE'] = '0'

# Disable numba caching which can cause issues in containers
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Disable numba completely to avoid JIT compilation issues
import numba
numba.config.DISABLE_JIT = True

# Set numba to use object mode to avoid JIT issues
numba.config.COMPATIBILITY_MODE = True

app = FastAPI()

# TODO: Move MongoDB credentials to environment variables for security
mongo_client = MongoClient("mongodb+srv://fakinwande:M50xQRyrwpnBGG9j@cluster0.f8wa06y.mongodb.net/")
db = mongo_client["environmental_sounds"]
retrain_collection = db["retrain_data"]
models_collection = db["models"]

# Function to clean up old data to manage storage
def cleanup_old_data():
    try:
        # Keep only the latest 3 model versions
        latest_models = list(models_collection.find().sort("timestamp", -1).limit(3))
        if len(latest_models) >= 3:
            # Delete older models
            oldest_timestamp = latest_models[-1]["timestamp"]
            models_collection.delete_many({"timestamp": {"$lt": oldest_timestamp}})
            print(f"Cleaned up old model versions")
        # Keep only recent retrain data (last 1000 records)
        retrain_count = retrain_collection.count_documents({})
        if retrain_count > 1000:
            # Delete oldest retrain data
            oldest_records = list(retrain_collection.find().sort("upload_timestamp", 1).limit(retrain_count - 1000))
            if oldest_records:
                oldest_timestamp = oldest_records[-1]["upload_timestamp"]
                retrain_collection.delete_many({"upload_timestamp": {"$lt": oldest_timestamp}})
                print(f"Cleaned up old retrain data, kept {1000} recent records")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Function to load latest model from MongoDB (for retraining)
def load_latest_model():
    try:
        # Get the most recent model from MongoDB
        latest_model = models_collection.find_one(
            sort=[("timestamp", -1)]
        )
        if latest_model:
            # Load model from bytes using temporary file
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                temp_model_path = temp_file.name
                temp_file.write(latest_model["model_data"])
            try:
                model = load_model(temp_model_path)
            finally:
                try:
                    os.remove(temp_model_path)
                except:
                    pass
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

# Function to load original model from filesystem (for prediction)
def load_original_model():
    try:
        model = load_model("audio_classifier_model.h5")
        le = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        print("Loaded original model from filesystem")
        print("Label classes:", le.classes_)
        return model, le, scaler
    except Exception as e:
        print(f"Error loading original model: {e}")
        return None, None, None

# Load original model & encoders for prediction
model, le, scaler = load_original_model()

RETRAIN_DIR = "retrain_data"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Feature Extraction ----------
def extract_features(file_path, n_mfcc=13, n_chroma=12, n_mel=128):
    try:
        # Load audio with minimal processing
        y, sr = librosa.load(file_path, sr=22050)
        if len(y) == 0:
            return None
        # Simple normalization
        y = y / (np.max(np.abs(y)) + 1e-6)
        features = []
        # MFCC features (most reliable)
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            features.extend([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.max(mfcc, axis=1), np.min(mfcc, axis=1)])
        except:
            dummy_mfcc = np.zeros(n_mfcc * 4)
            features.append(dummy_mfcc)
        # Mel spectrogram features
        try:
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.extend([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])
        except:
            dummy_mel = np.zeros(n_mel * 2)
            features.append(dummy_mel)
        # Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
            features.extend([np.mean(chroma, axis=1), np.std(chroma, axis=1)])
        except:
            dummy_chroma = np.zeros(n_chroma * 2)
            features.append(dummy_chroma)
        # Simple spectral features (avoiding numba-heavy functions)
        try:
            freqs = np.fft.fftfreq(len(y), 1/sr)
            fft_vals = np.abs(np.fft.fft(y))
            spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
            spectral_centroid_std = np.std(spectral_centroid) if hasattr(spectral_centroid, 'iter') else 0
            zero_crossings = np.sum(np.diff(np.signbit(y)))
            zero_crossing_rate = zero_crossings / len(y)
            features.extend([spectral_centroid, spectral_centroid_std, zero_crossing_rate, 0])
        except:
            features.extend([0, 0, 0, 0])
        feature_vector = np.concatenate([np.array(f).flatten() for f in features])
        if len(feature_vector) < 340:
            padding = np.zeros(340 - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > 340:
            feature_vector = feature_vector[:340]
        return feature_vector
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        print(f"Returning dummy features of size 340")
        return np.zeros(340)

# ---------- Predict Endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or le is None or scaler is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded properly."})
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(await file.read())
    try:
        features = extract_features(temp_filename)
    finally:
        try:
            os.remove(temp_filename)
        except:
            pass
    if features is None:
        return JSONResponse(status_code=400, content={"error": "Invalid audio file."})
    scaled = scaler.transform([features])
    probs = model.predict(scaled)
    pred_index = np.argmax(probs, axis=1)[0]
    pred_label = le.inverse_transform([pred_index])[0]
    print(f"Feature vector shape: {features.shape}")
    print(f"Scaled features shape: {scaled.shape}")
    print(f"Prediction probabilities: {probs[0]}")
    print(f"Predicted index: {pred_index}")
    print(f"Predicted label: {pred_label}")
    print(f"All class probabilities: {dict(zip(le.classes_, probs[0]))}")
    return {
        "prediction": pred_label,
        "confidence": float(probs[0][pred_index]),
        "all_probabilities": dict(zip(le.classes_, probs[0].tolist()))
    }

# ---------- Retrain Endpoint (Upload ZIP + Retrain) ----------
@app.post("/retrain")
async def retrain_model(zipfile_data: UploadFile = File(...)):
    try:
        print("Starting model retraining process...")
        
        # Step 1: Extract and validate ZIP file
        zip_content = await zipfile_data.read()
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                return {"error": "No CSV file found in the uploaded data."}
            
            csv_content = zip_ref.read(csv_files[0])
            df = pd.read_csv(io.BytesIO(csv_content))
            
            # Find label column
            possible_label_columns = ['label', 'category', 'class', 'target']
            label_column = None
            for col in possible_label_columns:
                if col in df.columns:
                    label_column = col
                    break
            
            if label_column is None:
                return {"error": f"No label column found. Available columns: {list(df.columns)}"}
            
            print(f"Found {len(df)} records in CSV file")
            
            # Step 2: Extract features from audio files
            features, labels = [], []
            processed_files = 0
            total_files = len(df)
            
            for _, row in df.iterrows():
                file = row['filename']
                label = row[label_column]
                
                if file in zip_ref.namelist() and file.endswith(".wav"):
                    audio_content = zip_ref.read(file)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_audio_path = temp_file.name
                        temp_file.write(audio_content)
                    
                    try:
                        fvec = extract_features(temp_audio_path)
                        if fvec is not None:
                            features.append(fvec)
                            labels.append(label)
                            
                            # Store in database
                            retrain_collection.insert_one({
                                "filename": file,
                                "label": label,
                                "features": fvec.tolist(),
                                "audio_data": audio_content,
                                "upload_timestamp": datetime.datetime.now(),
                                "metadata": row.to_dict()
                            })
                            processed_files += 1
                    finally:
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
            
            print(f"Successfully processed {processed_files}/{total_files} audio files")
        
        if len(features) < 2:
            return {"error": f"Not enough valid data to retrain. Only {len(features)} valid samples found."}
        
        # Step 3: Prepare data for training
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Unique labels: {np.unique(y)}")
        
        le_new = LabelEncoder()
        y_encoded = le_new.fit_transform(y)
        y_cat = to_categorical(y_encoded)
        scaler_new = StandardScaler()
        X_scaled = scaler_new.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size=0.2, stratify=y_encoded)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Step 4: Data augmentation
        print("Applying data augmentation...")
        X_train_augmented = []
        y_train_augmented = []
        
        for i in range(len(X_train)):
            # Original sample
            X_train_augmented.append(X_train[i])
            y_train_augmented.append(y_train[i])
            
            # Augmented sample 1: Low noise
            noise1 = np.random.normal(0, 0.01, X_train[i].shape)
            X_train_augmented.append(X_train[i] + noise1)
            y_train_augmented.append(y_train[i])
            
            # Augmented sample 2: Medium noise
            noise2 = np.random.normal(0, 0.02, X_train[i].shape)
            X_train_augmented.append(X_train[i] + noise2)
            y_train_augmented.append(y_train[i])
        
        X_train = np.array(X_train_augmented)
        y_train = np.array(y_train_augmented)
        
        print(f"Training data augmented from {len(X_scaled)} to {len(X_train)} samples")
        
        # Step 5: Load base model and create new model for transfer learning
        print("Loading base model for transfer learning...")
        base_model = load_model("audio_classifier_model.h5")
        
        # Create new model using transfer learning
        model_new = Sequential()
        for layer in base_model.layers[:-1]:
            model_new.add(layer)
        model_new.add(Dense(len(le_new.classes_), activation='softmax'))
        
        # Configure transfer learning: freeze base layers, train only new output layer
        print("Configuring transfer learning...")
        for layer in model_new.layers[:-1]:
            layer.trainable = False  # Freeze base model layers
        model_new.layers[-1].trainable = True  # Train only the new output layer
        
        # Single training phase with transfer learning
        print("Starting single-phase training with transfer learning...")
        optimizer = Adam(learning_rate=0.001)  # Moderate learning rate for transfer learning
        model_new.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Callbacks for better training
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        
        # Single training phase
        history = model_new.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=50,  # Reduced epochs for single phase
            batch_size=16,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        
        # Extract training metrics
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        print(f"Training completed - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Final metrics - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Step 6: Evaluate final model
        print("Evaluating final model...")
        overall_loss, overall_accuracy = model_new.evaluate(X_scaled, y_cat, verbose=0)
        
        # Step 7: Save model to database
        print("Saving model to database...")
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_model_path = temp_file.name
        
        model_new.save(temp_model_path)
        
        with open(temp_model_path, 'rb') as f:
            model_bytes = f.read()
        
        try:
            os.remove(temp_model_path)
        except:
            pass
        
        le_bytes = pickle.dumps(le_new)
        scaler_bytes = pickle.dumps(scaler_new)
        
        model_doc = {
            "timestamp": datetime.datetime.now(),
            "model_data": model_bytes,
            "label_encoder": le_bytes,
            "scaler": scaler_bytes,
            "accuracy": float(overall_accuracy),
            "loss": float(overall_loss),
            "classes": le_new.classes_.tolist(),
            "version": timestamp,
            "training_info": {
                "transfer_learning": {
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "epochs_trained": len(history.history['accuracy']),
                    "final_epoch": len(history.history['accuracy'])
                },
                "final": {
                    "overall_accuracy": float(overall_accuracy),
                    "overall_loss": float(overall_loss)
                }
            }
        }
        
        cleanup_old_data()
        models_collection.insert_one(model_doc)
        
        print("Model retraining completed successfully!")
        
        return {
            "message": "Model retrained and updated successfully using transfer learning.",
            "training_summary": {
                "processed_files": processed_files,
                "total_files": total_files,
                "valid_samples": len(features),
                "unique_classes": len(le_new.classes_),
                "classes": le_new.classes_.tolist(),
                "training_approach": "Single-phase transfer learning"
            },
            "transfer_learning_results": {
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epochs_trained": len(history.history['accuracy']),
                "learning_rate": 0.001,
                "base_model_layers_frozen": len(model_new.layers) - 1
            },
            "final_results": {
                "overall_accuracy": float(overall_accuracy),
                "overall_loss": float(overall_loss)
            }
        }
        
    except Exception as e:
        print(f"Error during retraining: {str(e)}")
        return {"error": f"Retraining failed: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Environmental Sounds API is running."}

@app.post("/cleanup")
async def cleanup_database():
    """Manually trigger database cleanup to free up space"""
    try:
        cleanup_old_data()
        return {"message": "Database cleanup completed successfully"}
    except Exception as e:
        return {"error": f"Cleanup failed: {str(e)}"}

@app.get("/storage")
async def get_storage_info():
    """Get current storage usage information"""
    try:
        model_count = models_collection.count_documents({})
        retrain_count = retrain_collection.count_documents({})
        latest_model = models_collection.find_one(sort=[("timestamp", -1)])
        latest_model_info = None
        if latest_model:
            latest_model_info = {
                "version": latest_model.get("version"),
                "timestamp": latest_model.get("timestamp"),
                "accuracy": latest_model.get("accuracy")
            }
        return {
            "model_versions": model_count,
            "retrain_records": retrain_count,
            "latest_model": latest_model_info,
            "message": "Consider using /cleanup endpoint if storage is high"
        }
    except Exception as e:
        return {"error": f"Failed to get storage info: {str(e)}"}

@app.get("/test-model")
async def test_model():
    """Test the model with dummy features to check if it's working"""
    try:
        if model is None or le is None or scaler is None:
            return {"error": "Model not loaded properly"}
        dummy_features = np.zeros(340)
        scaled = scaler.transform([dummy_features])
        probs = model.predict(scaled)
        pred_index = np.argmax(probs, axis=1)[0]
        pred_label = le.inverse_transform([pred_index])[0]
        random_features = np.random.normal(0, 1, 340)
        scaled_random = scaler.transform([random_features])
        probs_random = model.predict(scaled_random)
        pred_index_random = np.argmax(probs_random, axis=1)[0]
        pred_label_random = le.inverse_transform([pred_index_random])[0]
        return {
            "model_loaded": True,
            "classes": le.classes_.tolist(),
            "dummy_prediction": pred_label,
            "dummy_probabilities": dict(zip(le.classes_, probs[0].tolist())),
            "random_prediction": pred_label_random,
            "random_probabilities": dict(zip(le.classes_, probs_random[0].tolist())),
            "feature_vector_size": len(dummy_features),
            "scaler_fitted": hasattr(scaler, 'mean_'),
            "model_summary": str(model.summary()) if hasattr(model, 'summary') else "No summary available"
        }
    except Exception as e:
        return {"error": f"Model test failed: {str(e)}"}