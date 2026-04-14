// Main JavaScript for Deepfake Detection System

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const videoFile = document.getElementById('videoFile');
    const uploadArea = document.getElementById('uploadArea');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const progressSection = document.getElementById('progressSection');
    const progressBar = document.getElementById('progressBar');
    const statusText = document.getElementById('statusText');
    const modelStatus = document.getElementById('modelStatus');
    const clientError = document.getElementById('clientError');
    let selectedFile = null;
    let modelReady = false;

    checkModelReadiness();

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    uploadArea.addEventListener('click', function(e) {
        // The label already handles opening the file dialog natively via its `for` attribute.
        // Prevent a second dialog from opening when the click bubbles up from the label.
        if (e.target.tagName === 'LABEL' || e.target.closest('label') || e.target.tagName === 'INPUT') return;
        videoFile.click();
    });

    // File input change
    videoFile.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Handle file selection
    function handleFileSelect(file) {
        hideError();

        // Validate by extension first because browser MIME values vary by OS/browser.
        const extension = file.name.includes('.')
            ? file.name.split('.').pop().toLowerCase()
            : '';
        const validExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

        if (!validExtensions.includes(extension)) {
            showError('Please select a valid video file (MP4, AVI, MOV, MKV, or WEBM).');
            return;
        }

        const validTypes = [
            'video/mp4',
            'video/webm',
            'video/x-msvideo',
            'video/quicktime',
            'video/x-matroska'
        ];

        // If browser provides MIME, ensure it looks like a video type we support.
        if (file.type && !validTypes.includes(file.type)) {
            showError('Unsupported video format. Please upload MP4, AVI, MOV, MKV, or WEBM.');
            return;
        }

        // Validate file size (max 500MB)
        const maxSize = 500 * 1024 * 1024; // 500MB in bytes
        if (file.size > maxSize) {
            showError('File size exceeds 500MB. Please select a smaller file.');
            return;
        }

        // Display file info
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        analyzeBtn.disabled = !modelReady;
        selectedFile = file;
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    // Form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        hideError();

        if (!selectedFile) {
            showError('Please select a video file first.');
            return;
        }

        if (!modelReady) {
            showError('Model is not ready. Please load a trained checkpoint before running detection.');
            return;
        }

        // Disable button and show progress
        analyzeBtn.disabled = true;
        progressSection.style.display = 'block';
        updateProgress(10, 'Uploading video...');

        try {
            // Create FormData
            const formData = new FormData();
            formData.append('video', selectedFile);

            // Upload video
            updateProgress(30, 'Uploading video...');
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const uploadData = await uploadResponse.json();
            const jobId = uploadData.job_id;

            updateProgress(50, 'Processing video...');

            // Analyze video
            const analyzeResponse = await fetch(`/analyze/${jobId}`);

            if (!analyzeResponse.ok) {
                const errorData = await analyzeResponse.json();
                throw new Error(errorData.error || 'Analysis failed');
            }

            updateProgress(80, 'Generating results...');

            const analyzeData = await analyzeResponse.json();

            updateProgress(100, 'Complete! Redirecting...');

            // Redirect to results page
            setTimeout(() => {
                window.location.href = `/results/${jobId}`;
            }, 1000);

        } catch (error) {
            console.error('Error:', error);
            showError('Error: ' + error.message);
            
            // Reset UI
            analyzeBtn.disabled = !selectedFile || !modelReady;
            progressSection.style.display = 'none';
            updateProgress(0, '');
        }
    });

    async function checkModelReadiness() {
        try {
            const response = await fetch('/health');
            if (!response.ok) {
                throw new Error('Unable to reach health endpoint');
            }

            const health = await response.json();
            modelReady = Boolean(health.detector_ready);
            updateModelStatus(modelReady);
        } catch (error) {
            modelReady = false;
            updateModelStatus(false, 'Unable to verify model readiness. Ensure the server is running.');
        }

        analyzeBtn.disabled = !selectedFile || !modelReady;
    }

    function updateModelStatus(ready, customMessage = null) {
        if (ready) {
            modelStatus.className = 'alert alert-success';
            modelStatus.innerHTML = '<i class="fas fa-check-circle"></i> Model is ready for deepfake analysis.';
            return;
        }

        modelStatus.className = 'alert alert-warning';
        modelStatus.innerHTML = customMessage
            ? `<i class="fas fa-exclamation-triangle"></i> ${customMessage}`
            : '<i class="fas fa-exclamation-triangle"></i> Model checkpoint not found. Add trained weights to enable analysis.';
    }

    function showError(message) {
        clientError.textContent = message;
        clientError.style.display = 'block';
    }

    function hideError() {
        clientError.style.display = 'none';
        clientError.textContent = '';
    }

    // Update progress bar
    function updateProgress(percent, status) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = percent + '%';
        statusText.textContent = status;
    }
});

// Print functionality for results page
function printResults() {
    window.print();
}

// Download report (placeholder)
function downloadReport() {
    alert('Report download functionality coming soon!');
}
