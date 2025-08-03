# Environmental Sound Classification Web Application

**Project Description**

This project is an AI-powered web application designed to detect and classify environmental sounds using advanced audio feature extraction and a retrainable deep learning model. The system supports real-time prediction, retraining with new data, data visualizations, and performance monitoring — all integrated into an intuitive FastAPI backend.

**Key features of this application include:**

1. **Environmental Sound Prediction**  
   Users can upload `.wav` audio files, and the system will predict the sound class (e.g., rain, siren, engine) using a pre-trained deep neural network.

2. **Model Retraining**  
   A ZIP file containing new labeled audio data and metadata can be uploaded to retrain the model. The model is updated, versioned, and saved automatically.

3. **Data Visualizations**  
   The app includes visual plots such as waveform, spectrogram, and class distribution to help understand the acoustic characteristics of the dataset.

4. **Evaluation Metrics**  
   After retraining, the system reports accuracy, loss, and class-wise prediction probabilities.
   
5. **Performance Testing**  
   A Locust-based flood request simulation was used to test how the system handles high user load and concurrency.


Built with FastAPI, this backend service is integrated with MongoDB Atlas for data persistence, model versioning, and retrain data storage.

---

**Dataset Overview**

- **Name**: ESC-50 Environmental Sound Subset  
- **Source**: [Click here to access dataset](<https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50?select=audio>)    
- **Description**:  
  The dataset i used is a curated subset of the original ESC-50 dataset, specifically filtered to include only **environmental sounds** that are relevant to real-world ambient and urban environments. The dataset comprises **16 distinct classes** with **40 audio samples per class**, each 5 seconds long.
  
- **Why These Classes?**  
The selected sound classes were chosen based on their relevance to natural and human-made environmental conditions, making them ideal for training a model that can interpret acoustic environments. These sounds reflect a diverse blend of **natural phenomena**, **urban noise**, and **warning signals**.

- **Selected Classes**:
  - **`rain`**
  - **`thunderstorm`**
  - **`sea_waves`**
  - **`water_drops`**
  - **`pouring_water`**
  - **`wind`**
  - **`crickets`**
  - **`chirping_birds`**
  - **`crackling_fire`**
  - **`glass_breaking`**
  - **`engine`**
  - **`car_horn`**
  - **`train`**
  - **`airplane`**
  - **`siren`**
  - **`toilet_flush`**


- **Feature Extraction**:  
  Audio files were processed using Librosa to extract:
  - **MFCCs** (Mel Frequency Cepstral Coefficients)
  - **Chroma STFT**
  - **Mel Spectrogram**
  - **Spectral Centroid, Rolloff, Bandwidth, Zero Crossing Rate**
  These features were normalized and used as input to a deep neural network classifier.

- **Target**:  
  `class` — A categorical label representing one of the 16 environmental sound classes.

---

**Demo Video:** [Click here to access dataset](<https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50?select=audio>)  

**Live Api Endpoint:** [Click here to access API endpoint ](<https://lollypopping-environmental-sounds.hf.space>)


---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/FavourAkinwande/Environmental_Sounds_MLops.git
cd Environmental_Sounds_MLops
```

### **2. Set Up and Run the Application Locally**

#### **Create a Virtual Environment**
```bash
python -m venv venv
```

#### **Activate the Virtual Environment**
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **Run the Application**
```bash
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

---

## **Features Overview**

### **1. Environmental Sound Prediction**
- **Endpoint:** `/predict`
- **Input:** `.wav` audio file
- **Output:** Predicted sound class and confidence score

### **2. Model Retraining**
- **Endpoint:** `/retrain`
- **Input:** ZIP file containing labeled audio files and a metadata CSV
- **Output:**  
  - Updated model version saved  
  - Evaluation metrics (accuracy, loss)

### **3. Data Visualizations**
- **Page URL (UI is integrated):** `/visualizations`
- **Features:** Visual representations of:
  - Waveforms
  - Spectrograms
  - Audio durations
  - Class distributions 
---

## **Flood Request Simulation**

The system’s performance was evaluated under heavy user load using **Locust**, simulating concurrent requests to test responsiveness and throughput.

- **Average response time:** *X seconds*
- **Peak concurrent users handled:** *Y users*

Detailed test results:

- **CSV Report:** [Locust Simulation Results](/locust.csv)
- **Visualization:**  
  ![Flood Simulation Chart](/total_requests_per_second.png)

---

## **Results and Evaluation**

### **Prediction Example**
| Metric           | Value     |
|------------------|-----------|
| Predicted Class  | Engine      |
| Confidence Score | %     |

---

## **Contributing**

Contributions are welcome!  
Feel free to fork the repository, submit issues, or open pull requests to improve features and performance.

---

## **License**

This project is licensed under the **MIT License**. See the LICENSE file for details.

---
