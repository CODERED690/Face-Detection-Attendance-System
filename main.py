from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "CODE RED: FastAPI backend"}

@app.get("/generate_dataset/{path}")
async def gen(path: str):
    from generate_dataset import generate
    return StreamingResponse(generate(path), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/train_model")
async def tra():
    from train_model import train
    return {"message": train()}

@app.get("/recognize_face")
async def rec():
    from recognize_face import display
    return StreamingResponse(display(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)