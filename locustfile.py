from locust import HttpUser, task, between
import time

class Fastapi(HttpUser):
    wait_time = between(2, 5)
    host = "https://lollypopping-environmental-sounds.hf.space"

    @task
    def make_prediction(self):
        # You must pass a real file to /predict
        with open("sample.wav", "rb") as f:
            files = {"file": ("sample.wav", f, "audio/wav")}
            self.client.post("/predict", files=files)

    @task
    def retrain(self):
        # You must pass a real zip file to /retrain
        with open("retrain_data.zip", "rb") as f:
            files = {"zipfile_data": ("retrain_data.zip", f, "application/zip")}
            self.client.post("/retrain", files=files)
