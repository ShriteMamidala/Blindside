from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

@app.get("/")
async def root():
    return {"message": "Welcome to Blindside Backend"}

# WebSocket for blind person
@app.websocket("/ws/blind")
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

# WebSocket for friends/family
@app.websocket("/ws/blind")
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