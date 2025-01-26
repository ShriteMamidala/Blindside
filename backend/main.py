from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Blindside Backend"}

# WebSocket for blind person
@app.websocket("/ws/blind")
async def websocket_endpoint_blind(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket for blind person connected")
    while True:
        try:
            data = await websocket.receive_text()
            print(f"Received from blind person: {data}")
            await websocket.send_text("Processing your input...")
        except Exception as e:
            print(f"Connection closed: {e}")
            break

# WebSocket for friends/family
@app.websocket("/ws/family")
async def websocket_endpoint_family(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket for family/friends connected")
    while True:
        try:
            data = await websocket.receive_text()
            print(f"Received from family/friends: {data}")
        except Exception as e:
            print(f"Connection closed: {e}")
            break