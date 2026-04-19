import cv2
import numpy as np
import pickle
import base64
import mediapipe as mp
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Setup templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
try:
    with open('model/sign_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Setup mediapipe HandLandmarker (IMAGE mode for stateless API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error loading Mediapipe model: {e}")
    landmarker = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(image: str = Form(...)):
    if not model or not landmarker:
        return JSONResponse(content={"error": "Models not loaded correctly on server"}, status_code=500)
        
    try:
        # Decode base64 image
        header, encoded = image.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Mediapipe requires RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect
        result = landmarker.detect(mp_image)
        
        if result and result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            
            # Extract landmarks for prediction same as app.py
            landmarks = []
            for lm in hand_landmarks:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks_array = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks_array)[0]
            probabilities = model.predict_proba(landmarks_array)[0]
            confidence = float(max(probabilities) * 100)
            
            return JSONResponse(content={"prediction": prediction, "confidence": confidence})
        
        return JSONResponse(content={"prediction": None, "confidence": 0})
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
