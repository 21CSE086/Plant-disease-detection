from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

# Define CNN Model
class TomatoDiseaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(TomatoDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = TomatoDiseaseCNN()
model.load_state_dict(torch.load("tomato_disease_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define class labels
class_labels = ["Healthy", "Early Blight", "Late Blight", "Septoria Leaf Spot", "Bacterial Spot"]

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Return response
        return JSONResponse(content={
            "prediction": class_labels[predicted_class.item()],
            "confidence": confidence.item() * 100
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run server with: uvicorn backend:app --reload