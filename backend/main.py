from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from pprint import pprint


app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

model = YOLO(r'runs\detect\train4\weights\best.pt')  # Path to trained YOLOv8 weights
model.eval()

@app.get("/")
async def root():
    return {"message": "Welcome to Blindside Backend"}

@app.websocket("/ws/blind")
async def websocket_endpoint_blind(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive binary frame data
            frame_data = await websocket.receive_bytes()

            # Preprocess frame data into a format compatible with YOLO
            input_image = preprocess_frame_data(frame_data)

            # Run YOLO inference
            results = model.predict(input_image, imgsz=640, conf=0.5)

            # Extract and pprint the detected class names
            detected_classes = []
            for result in results:
                # YOLOv8 `result` object contains `.names` and `.boxes.cls`
                detected_classes = [model.names[int(cls)] for cls in result.boxes.cls]
            pprint(detected_classes)  # Pretty-print the detected classes

            # Send back a message to the client
            await websocket.send_text("Frame processed")
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected with code: {e.code}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

def preprocess_frame_data(frame_data):
    """
    Convert binary frame data to a format compatible with YOLOv8.
    """
    # Load image from binary data
    image = Image.open(io.BytesIO(frame_data)).convert("RGB")

    # Convert image to numpy array
    image_np = np.array(image)

    # Return the image (YOLOv8 can process numpy arrays directly)
    return image_np


# WebSocket for friends/family
@app.websocket("/ws/friend_family")
async def websocket_endpoint_blind(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Assuming you are expecting to receive binary data
            frame_data = await websocket.receive_bytes()
            # Process received frame data here...
            await websocket.send_text("Frame received")
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected with code: {e.code}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        # Safely close the websocket if not already closed
        if not websocket.application_state == "CLOSED":
            await websocket.close()