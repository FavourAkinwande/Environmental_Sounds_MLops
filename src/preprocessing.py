from sklearn.preprocessing import StandardScaler

#Patch deprecated np.complex for compatibility
np.complex = complex

# Enhanced feature extraction function
def extract_features(file_path, n_mfcc=13, n_chroma=12, n_mel=128):
    """Extract comprehensive audio features"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=22050)
        y, _ = librosa.effects.trim(y)
        
        # Normalize
        if len(y) > 0:
            y = y / (np.max(np.abs(y)) + 1e-6)
        else:
            return None
            
        # Extract multiple features
        features = []
        
        # 1. MFCCs (13 coefficients, keep temporal info)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features.extend([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.max(mfcc, axis=1),
            np.min(mfcc, axis=1)
        ])
        
        # 2. Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        features.extend([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ])
        
        # 3. Mel-scale spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mel)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend([
            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1)
        ])
        
        # 4. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        features.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        # Flatten all features
        return np.concatenate([np.array(f).flatten() for f in features])
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Extract features for all files
print("Extracting enhanced features...")
features = []
labels = []

for idx, row in env_df.iterrows():
    file_path = os.path.join(AUDIO_DIR, row['filename'])
    feature_vector = extract_features(file_path)
    
    if feature_vector is not None:
        features.append(feature_vector)
        labels.append(row['category'])

# Convert to arrays
X = np.array(features)
y = np.array(labels)

print(f"Feature extraction complete. Shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded  # Ensure balanced split
)

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of classes: {num_classes}")
print(f"Classes: {le.classes_}")
