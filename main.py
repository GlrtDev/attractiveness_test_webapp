import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from predictor import predictor
from preprocess import preprocessor
import cv2
import base64

app = FastAPI()
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile/")
async def upload_file(file: UploadFile, request: Request):
    contents = await file.read()
    img = preprocessor.load_image(contents)
    rating = predictor.predict(img)
    ret, img_buff = cv2.imencode('.png', img) #could be png, update html as well
    img_b64 = base64.b64encode(img_buff)
    return templates.TemplateResponse("rating.html", {"request": request, "rating": rating[0], "img_bytes": img_b64})


if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info")