from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Blindside Backend!"}