document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.textContent = 'Predicting...';
    const fileInput = document.getElementById('audio-file');
    if (!fileInput.files.length) {
        resultDiv.textContent = 'Please select a WAV file.';
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
        const response = await fetch('https://lollypopping-environmental-sounds.hf.space/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            resultDiv.innerHTML = `<strong>Prediction:</strong> ${data.prediction}<br><strong>Confidence:</strong> ${data.confidence.toFixed(3)}`;
        } else {
            resultDiv.textContent = data.error || 'Prediction failed.';
        }
    } catch (err) {
        resultDiv.textContent = 'Error connecting to prediction service.';
    }
});