import torch
model = torch.load("tomato_disease_model.pth", map_location=torch.device('cpu'))
print(model)
