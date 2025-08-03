document.getElementById('retrain-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultDiv = document.getElementById('retrain-result');
    const fileInput = document.getElementById('zip-file');
    const trainButton = document.querySelector('.upload-btn');
    
    // Check if file is selected
    if (!fileInput.files.length) {
        resultDiv.innerHTML = '<div class="error-message">⚠️ Please select a ZIP file to upload.</div>';
        return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.zip')) {
        resultDiv.innerHTML = '<div class="error-message">⚠️ Please upload a ZIP file only.</div>';
        return;
    }
    
    // Show loading state and disable button
    resultDiv.innerHTML = '<div class="loading-message">🔄 Sending dataset to API... Please wait.</div>';
    trainButton.disabled = true;
    trainButton.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2v6m0 0l-3-3m3 3l3-3" stroke="currentColor" stroke-width="2" fill="none"/><path d="M12 8v8" stroke="currentColor" stroke-width="2" fill="none"/></svg> Training...';
    
    // Create FormData and append file
    const formData = new FormData();
    formData.append('zipfile_data', file);
    
    try {
        // Send to API endpoint
        const response = await fetch('https://lollypopping-environmental-sounds.hf.space/retrain', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Success - display only the message
            resultDiv.innerHTML = `
                <div class="success-message">
                    <h3>✅ ${data.message || 'Model retrained successfully!'}</h3>
                </div>
            `;
        } else {
            // API error
            resultDiv.innerHTML = `<div class="error-message">❌ ${data.error || 'Retraining failed. Please try again.'}</div>`;
        }
    } catch (err) {
        // Network or connection error
        console.error('Retraining error:', err);
        resultDiv.innerHTML = '<div class="error-message">❌ Error connecting to retraining service. Please check your internet connection and try again.</div>';
    } finally {
        // Re-enable button
        trainButton.disabled = false;
        trainButton.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2v6m0 0l-3-3m3 3l3-3" stroke="currentColor" stroke-width="2" fill="none"/><path d="M12 8v8" stroke="currentColor" stroke-width="2" fill="none"/></svg> Start Training';
    }
});

// Add file name display functionality
const fileInput = document.getElementById('zip-file');
const uploadArea = document.querySelector('.upload-area');

// Function to update file name display
function updateFileName() {
    const file = fileInput.files[0];
    const uploadText = uploadArea.querySelector('h3');
    const uploadDesc = uploadArea.querySelector('p');
    
    if (file) {
        // Show file name
        uploadText.textContent = `📦 ${file.name}`;
        uploadDesc.textContent = 'Dataset selected! Click "Start Training" to begin retraining.';
        
        // Add visual feedback
        uploadArea.style.borderColor = '#388e3c';
        uploadArea.style.background = '#f8f9fa';
    } else {
        // Reset to default
        uploadText.textContent = 'Click to Upload Dataset';
        uploadDesc.textContent = 'Drag and drop your ZIP file here or click to browse';
        
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

// Add click functionality to upload area (icon and text, not button)
uploadArea.addEventListener('click', function(e) {
    // Don't trigger if clicking on the button or form elements
    if (e.target.tagName === 'BUTTON' || 
        e.target.closest('button') || 
        e.target.tagName === 'INPUT' || 
        e.target.closest('form')) {
        return; // Don't trigger file input for button/form clicks
    }
    fileInput.click();
});