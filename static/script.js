const video = document.getElementById('webcam');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const predictionDisplay = document.getElementById('predictionDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const toggleBtn = document.getElementById('toggleBtn');

let isRunning = false;
let stream = null;
let captureInterval = null;

// Used to grab frames
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        });
        video.srcObject = stream;
        isRunning = true;
        statusText.innerText = "Camera Active";
        statusBadge.style.color = "var(--success)";
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        // Start sending frames 3 times per second
        captureInterval = setInterval(sendFrameToBackend, 333); 
        
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusText.innerText = "Camera Access Denied";
        statusBadge.style.color = "#ef4444"; // Red error
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    clearInterval(captureInterval);
    isRunning = false;
    video.srcObject = null;
    statusText.innerText = "Camera Paused";
    statusBadge.style.color = "var(--text-secondary)";
}

toggleBtn.addEventListener('click', () => {
    if (isRunning) {
        stopWebcam();
        toggleBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
        predictionDisplay.innerText = "--";
        confidenceDisplay.innerText = "Camera is paused";
    } else {
        startWebcam();
        toggleBtn.innerHTML = '<i class="fas fa-video-slash"></i> Stop Camera';
        predictionDisplay.innerText = "--";
        confidenceDisplay.innerText = "Analyzing frame...";
    }
});

// A rolling history to stabilize UI (similar to app.py)
let predictionHistory = [];
const HISTORY_LENGTH = 5;

async function sendFrameToBackend() {
    if (!isRunning || !canvas.width || !canvas.height) return;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64 jpeg
    const base64Image = canvas.toDataURL('image/jpeg', 0.8);

    const formData = new FormData();
    formData.append('image', base64Image);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.prediction && data.prediction !== "None") {
            // Stabilize predictions
            predictionHistory.push(data.prediction);
            if(predictionHistory.length > HISTORY_LENGTH) {
                predictionHistory.shift();
            }
            
            // Get most frequent prediction in history
            const counts = predictionHistory.reduce((acc, val) => {
                acc[val] = (acc[val] || 0) + 1;
                return acc;
            }, {});
            const currentPrediction = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);

            predictionDisplay.innerText = currentPrediction;
            confidenceDisplay.innerText = `Confidence: ${data.confidence.toFixed(1)}%`;
            confidenceDisplay.style.color = "white";
        } else {
            predictionDisplay.innerText = "--";
            confidenceDisplay.innerText = "No hand detected";
            confidenceDisplay.style.color = "var(--text-secondary)";
        }
        
    } catch (err) {
        console.error("Error from backend:", err);
    }
}

// initialize
startWebcam();
