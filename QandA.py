#!/usr/bin/env python3
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import cv2

# prepare image + question
# url = "https://womensagenda.com.au/wp-content/uploads/2023/09/Screenshot-2023-09-12-at-2.11.54-PM-1024x932.png"
# image = Image.open(requests.get(url, stream=True).raw)


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

while True:
    image = cv2.imread(input("path: "))
    text = input("> ")
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])

