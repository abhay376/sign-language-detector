# 🤟 Live Sign Language Detector 

An interactive, real-time web application that leverages computer vision and machine learning to detect and classify American Sign Language (ASL) gestures directly via your browser's webcam.


---

## 🧠 Architecture & How it Works (For Interviews)

This project was built to transform a standard local python script into a full-stack, production-ready web application. It is broken down into three main systems:

### 1. The Frontend (Vanilla JS & HTML5)
- **WebRTC & Canvas:** The app requests native browser access to the user's webcam and captures live video. We use an HTML5 `<canvas>` to grab image snapshot frames from the live video feed.
- **REST API Interaction:** Instead of heavy data-streaming, the Javascript uses the `Fetch API` to send lightweight `Base64` encoded JPEG images over to the Backend three times a second. 
- **Dynamic UI:** Upon receiving the AI prediction, the DOM is updated asynchronously, showing the result and the model confidence string without ever reloading the page. 

### 2. The Backend Engine (FastAPI)
- **High Performance API:** The backend uses `FastAPI` to create an extremely fast, asynchronous python web server. FastAPI handles the incoming POST requests, unpacks the base64 image strings, and passes them to our ML models.
- **Stateless Design:** Unlike local scripts which keep a constant webcam stream open (`cv2.VideoCapture`), this backend is **stateless**. It receives an image, runs detection (`mp.tasks.vision.RunningMode.IMAGE`), gives an answer, and immediately frees up resources. This is how it successfully acts as a Web API.

### 3. The Machine Learning Pipeline
Once FastAPI has the image array, it goes through a dual-model pipeline:
1. **Google MediaPipe (Hand Landmarker):** Uses an optimized pre-trained model to detect a hand in the frame and locate 21 distinct 3D landmarks (knuckles, fingertips, etc.).
2. **Custom Scikit-Learn Model (Random Forest):** We take the exact coordinates of those 21 landmarks, flatten them into an array, and pass them into our custom-trained `sign_model.pkl`. This model then classifies the spatial relationships of the hand into a specific Sign Language letter!

---

## 🛠️ Technology Stack Breakdown
- **Backend:** `Python 3`, `FastAPI`, `Uvicorn`
- **Machine Learning Engine:** `Scikit-Learn` (RandomForest Model), `Google MediaPipe` (Hand Landmarker SDK)
- **Image Processing:** `OpenCV`, `NumPy`
- **Frontend UI:** `HTML5 Canvas`, Premium Dark Mode `CSS`, `Vanilla JS`
- **Deployment:** `Vercel` (Static Portfolio)

---

## 💻 Running Locally

To run this application on your own machine:

1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```
4. Open the displayed `localhost:8000` link in your web browser.
