from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from pprint import pprint
import time
import threading
from datetime import datetime, timedelta
from google.cloud import texttospeech
import os
from fastapi.responses import FileResponse
from playsound import playsound


app = FastAPI()

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
print(f"Resolved absolute path: {frontend_path}")
if not os.path.isdir(frontend_path):
    raise RuntimeError(f"Directory '{frontend_path}' does not exist")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")


current_dir = os.path.dirname(__file__)

# Navigate one directory up to remove 'backend' from the path
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Construct the path to the YOLO weights file
weights_path = os.path.join(project_root, r"runs\detect\train4\weights\best.pt")

# Load the model with the relative path
model = YOLO(weights_path)
model.eval()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_CLOUD_TTS"]


CLASS_PRIORITY = {
    "pedestrian": 4,
    "stop sign": 3,
    "crosswalk": 2,
    "car": 1
}

# Track last spoken time for each class
last_spoken_time = {}
lock = threading.Lock()

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
            results = model.predict(input_image, imgsz=640, conf=0.1)

            # Extract detections with bounding boxes and class names
            detections = []
            for result in results:
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    class_name = model.names[int(cls)]

                    # Convert tensors to NumPy arrays
                    bbox = box.cpu().numpy()  # Move bbox to CPU and convert to numpy
                    detections.append({"class": class_name, "bbox": bbox})

            # Process the detections based on hierarchy and location
            if detections:
                message = determine_audio_message(detections)
                if message:
                    # Trigger audio output (add text-to-speech logic here)
                    print(f"Audio Alert: {message}")
                    await websocket.send_text(message)

            # Send back a generic acknowledgment to the client
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


def determine_audio_message(detections):
    """
    Determine the audio message to be said based on hierarchy and location.
    """
    global last_spoken_time

    # Sort detections by hierarchy (descending)
    detections.sort(key=lambda x: CLASS_PRIORITY.get(x["class"], 0), reverse=True)

    for detection in detections:
        class_name = detection["class"]
        bbox = detection["bbox"]

        # Determine location based on bounding box
        location = determine_location(bbox)

        # Check the time since the last alert for this class
        current_time = datetime.now()
        with lock:
            last_time = last_spoken_time.get(class_name, None)
            if last_time and (current_time - last_time).total_seconds() < 10:
                continue  # Skip this detection if the time gap is less than 10 seconds

            # Update the last spoken time for this class
            last_spoken_time[class_name] = current_time

        # Generate audio message
        return f"{class_name} detected {location}"

    return None  # No valid detection to speak


def determine_location(bbox):
    """
    Determine the location of the object based on its bounding box.
    """
    x_center = (bbox[0] + bbox[2]) / 2  # Average of x_min and x_max
    if x_center < 0.33:  # Left-third of the frame
        return "on the left"
    elif x_center > 0.66:  # Right-third of the frame
        return "on the right"
    else:  # Center-third of the frame
        return "straight ahead"


def text_to_speech(message):
    """
    Convert text to speech using Google Cloud TTS and play it out loud.
    """
    # Initialize the TTS client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    input_text = texttospeech.SynthesisInput(text=message)

    # Set the voice parameters (language, voice type)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",  # Adjust for your language
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Set the audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    # Save the audio to a file
    audio_file = "output.mp3"
    with open(audio_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to '{audio_file}'")

    print(f"Audio file generated: {os.path.abspath('output.mp3')}")

    # Play the audio file
    playsound(audio_file)

    # Optionally, delete the file after playback
    os.remove(audio_file)


# Start a background thread for managing audio gaps (optional)
def audio_loop():
    while True:
        time.sleep(3)  # Ensure a 3-second gap between messages

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

@app.get("/audio")
async def get_audio():
    return FileResponse("output.mp3", media_type="audio/mpeg")