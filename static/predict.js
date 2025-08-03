document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultDiv = document.getElementById('prediction-result');
    const fileInput = document.getElementById('audio-file');
    const analyzeButton = document.querySelector('.upload-btn');
    
    // Check if file is selected
    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<div class="error-message">⚠️ Please select a WAV file to analyze.</div>';
        return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file type
    if (!file.type.includes('audio/wav') && !file.name.toLowerCase().endsWith('.wav')) {
        resultDiv.innerHTML = '<div class="error-message">⚠️ Please upload a WAV file only.</div>';
        return;
    }
    
    // Show loading state and disable button
    resultDiv.innerHTML = '<div class="loading-message">🔄 Sending file to API... Please wait.</div>';
    analyzeButton.disabled = true;
    analyzeButton.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2v6m0 0l-3-3m3 3l3-3" stroke="currentColor" stroke-width="2" fill="none"/><path d="M12 8v8" stroke="currentColor" stroke-width="2" fill="none"/></svg> Processing...';
    
    // Create FormData and append file
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Send to API endpoint
        const response = await fetch('https://lollypopping-environmental-sounds.hf.space/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Success - display results
            resultDiv.innerHTML = `
                <div class="success-message">
                    <h3>🎯 Analysis Complete!</h3>
                    <div class="prediction-result">
                        <p><strong>File:</strong> <span class="file-name">${file.name}</span></p>
                        <p><strong>Detected Sound:</strong> <span class="prediction-label">${data.prediction}</span></p>
                        <p><strong>Confidence:</strong> <span class="confidence-score">${(data.confidence * 100).toFixed(1)}%</span></p>
                    </div>
                </div>
            `;
        } else {
            // API error
            resultDiv.innerHTML = `<div class="error-message">❌ ${data.error || 'Prediction failed. Please try again.'}</div>`;
        }
    } catch (err) {
        // Network or connection error
        console.error('Prediction error:', err);
        resultDiv.innerHTML = '<div class="error-message">❌ Error connecting to prediction service. Please check your internet connection and try again.</div>';
    } finally {
        // Re-enable button
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 12l2 2 4-4" stroke="currentColor" stroke-width="2" fill="none"/><path d="M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9 9 4.03 9 9z" stroke="currentColor" stroke-width="2" fill="none"/></svg> Analyze Sound';
    }
});

// Add file name display functionality
const fileInput = document.getElementById('audio-file');
const uploadArea = document.querySelector('.upload-area');

// Function to update file name display
function updateFileName() {
    const file = fileInput.files[0];
    const uploadText = uploadArea.querySelector('h3');
    const uploadDesc = uploadArea.querySelector('p');
    
    if (file) {
        // Show file name
        uploadText.textContent = `📁 ${file.name}`;
        uploadDesc.textContent = 'File selected! Click "Analyze Sound" to process.';
        
        // Add visual feedback
        uploadArea.style.borderColor = '#388e3c';
        uploadArea.style.background = '#f8f9fa';
    } else {
        // Reset to default
        uploadText.textContent = 'Click to Upload Audio';
        uploadDesc.textContent = 'Drag and drop your WAV file here or click to browse';
        
        // Reset visual feedback
        uploadArea.style.borderColor = '#388e3c';
        uploadArea.style.background = '#fff';
    }
}

// Listen for file input changes
fileInput.addEventListener('change', updateFileName);

// Add drag and drop functionality
if (uploadArea && fileInput) {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            // Update file name display
            updateFileName();
        }
    }
}

// Add click functionality to upload area
uploadArea.addEventListener('click', function(e) {
    // Don't trigger if clicking on the button or form elements
    if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
        return; // Don't trigger file input for button clicks
    }
    fileInput.click();
});
