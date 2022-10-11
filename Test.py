from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image

image = Image.open(r'C:\Users\Public\idback.jpg').convert('RGB')

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 16 RVL-CDIP classes
predicted_class_idx = logits.argmax(-1).item()
print("Pre dicted class:", model.config.id2label[predicted_class_idx])