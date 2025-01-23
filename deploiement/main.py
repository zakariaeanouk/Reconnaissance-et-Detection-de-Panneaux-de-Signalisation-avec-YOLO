from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import torch
import numpy as np
import cv2
import io
import os
import time
import glob
from ultralytics import YOLO

# Charger les modèles
detection_model = YOLO('/Users/zakariaeanouk/Desktop/projet ML/detection/best.pt')
classification_model = YOLO('/Users/zakariaeanouk/Desktop/projet ML/best.pt')

def cleanup_static_folder(static_dir: str, keep_file: str = None):
    """
    Nettoie le dossier static en supprimant toutes les images annotées sauf celle spécifiée
    """
    pattern = os.path.join(static_dir, "annotated_image_*.jpg")
    for file_path in glob.glob(pattern):
        if keep_file is None or os.path.basename(file_path) != keep_file:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def process_detection_output(detection_output):
    boxes = detection_output[0].boxes.xyxy
    boxes = boxes.cpu().numpy()
    return boxes

def process_classification_output(classification_output):
    predictions = []
    for result in classification_output:
        class_idx = int(result.boxes[0].cls)
        class_name = result.names[class_idx]
        predictions.append(class_name)
    return predictions

def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), label, fill="red")
    return image

app = FastAPI()

# Ajouter le middleware CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# Nettoyer le dossier static au démarrage
cleanup_static_folder(static_dir)

# Monter le dossier statique
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_np = np.array(image)
        
        detections = detection_model.predict(image_np, imgsz=1024)
        
        boxes = process_detection_output(detections)
        if boxes.size > 0:
            predictions = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                roi = image_np[y1:y2, x1:x2]
                roi_resized = cv2.resize(roi, (640, 640))
                roi_tensor = torch.tensor(roi_resized).permute(2, 0, 1).unsqueeze(0).float()
                roi_tensor = roi_tensor / 255.0
                class_output = classification_model.predict(roi_tensor)
                prediction = process_classification_output(class_output)
                predictions.append(prediction[0])
            
            # Dessiner les boîtes et les labels sur l'image
            image_with_boxes = draw_boxes(image, boxes, predictions)
            
            # Redimensionner l'image avant de la sauvegarder
            max_size = (800, 800)
            image_with_boxes.thumbnail(max_size, Image.LANCZOS)
            
            # Generate a unique filename using timestamp
            image_filename = f"annotated_image_{int(time.time())}.jpg"
            image_path = os.path.join(static_dir, image_filename)
            
            # Nettoyer les anciennes images avant de sauvegarder la nouvelle
            cleanup_static_folder(static_dir, image_filename)
            
            # Sauvegarder l'image
            image_with_boxes.save(image_path)
            
            return JSONResponse(content={
                "predictions": predictions,
                "image_url": f"static/{image_filename}"
            })
        else:
            return JSONResponse(content={"message": "No detections found."})
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred during processing: {str(e)}"}
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)