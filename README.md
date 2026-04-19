# Live Sign Language Detector 🤟

A web application that leverages real-time computer vision and machine learning to detect and classify American Sign Language (ASL) gestures directly via your browser's webcam.

## 🔗 Live Application

The project has been configured to be deployed as a web service. 
**👉 [View the Live Application Here](https://sign-language-detector.onrender.com)**  
*(Note: Replace the link above with your actual deployment URL if Render assigned a slightly different domain)*

## 🛠️ Technology Stack
- **Backend:** Python, FastAPI, Uvicorn 
- **Machine Learning Engine:** Scikit-Learn (RandomForest Model), Google MediaPipe (Hand Landmarker), OpenCV
- **Frontend UI:** HTML5 Canvas, Vanilla CSS (Premium Dark Mode), Javascript / WebRTC

## 💻 Running Locally

To run this application on your own machine:

1. Clone this repository.
2. Ensure you have Python installed, then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI development server:
   ```bash
   uvicorn main:app --reload
   ```
4. Open the displayed `localhost:8000` link in your web browser.

## 🚀 Deployment (Render)

This repository includes a `render.yaml` blueprint.
To deploy it securely:
1. Go to [Render.com](https://render.com/)
2. Create a "New Blueprint" and connect this GitHub repo.
3. Your web app with native Python ML capabilities will be built and published securely!
