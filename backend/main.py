from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/images", StaticFiles(directory="dataset"))

@app.get("/")
async def root():
    return {"message": "CODE RED: FastAPI backend"}

@app.get("/attendance_list")
async def att():
    from send_attendance import attendance
    return JSONResponse(content=attendance(), headers={"Access-Control-Allow-Origin": "*"})

@app.get("/generate_dataset/{path}")
async def gen(path: str):
    from generate_dataset import generate
    return StreamingResponse(generate(path), media_type='multipart/x-mixed-replace; boundary=frame', headers={"Access-Control-Allow-Origin": "*"})

@app.get("/train_model")
async def tra():
    from train_model import train
    return {"message": train()}

@app.get("/recognize_face")
async def rec():
    from recognize_face import display
    return StreamingResponse(display(), media_type='multipart/x-mixed-replace; boundary=frame', headers={"Access-Control-Allow-Origin": "*"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6969)