document.getElementById('retrain-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('retrain-result');
    resultDiv.textContent = 'Retraining...';
    const fileInput = document.getElementById('zip-file');
    if (!fileInput.files.length) {
        resultDiv.textContent = 'Please select a ZIP file.';
        return;
    }
    const formData = new FormData();
    formData.append('zipfile_data', fileInput.files[0]); // field name must match FastAPI endpoint
    try {
        const response = await fetch('https://lollypopping-environmental-sounds.hf.space/retrain', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            resultDiv.innerHTML = `<strong>${data.message || 'Model retrained successfully.'}</strong>`;
        } else {
            resultDiv.textContent = data.error || 'Retrain failed.';
        }
    } catch (err) {
        resultDiv.textContent = 'Error connecting to retrain service.';
    }
});