from utils import *

import cv2
import torch
import time
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw

# Load SSD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoint_ssd300.pth.tar", weights_only=False)
model = checkpoint["model"].to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


cap = cv2.VideoCapture(0)

prev_time = time.time()

min_score=0.5

max_overlap=0.5

top_k=200

suppress=None



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))

    # Object detection
    """
    Detect objects in a frame using SSD.
    """
    original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Transform image for model input
    img_tensor = normalize(to_tensor(resize(original_image))).to(device)

    # Forward pass
    with torch.no_grad():
        predicted_locs, predicted_scores = model(img_tensor.unsqueeze(0))

        # Detect objects
        det_boxes, det_labels, det_scores = model.detect_objects(
            predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k
        )

    # Move boxes to CPU
    det_boxes = det_boxes[0].to("cpu")

    # Transform to original image size
    original_dims = torch.FloatTensor([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Convert PIL image to OpenCV format
    draw = ImageDraw.Draw(original_image)

    # Draw detections
    for i in range(det_boxes.size(0)):
        label = rev_label_map[det_labels[0][i].item()]
        

        box = det_boxes[i].numpy().astype(int)
        xmin, ymin, xmax, ymax = box

        # Draw rectangle
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Calculate center of the detected object
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Hedefe çizgi çiz
        cv2.line(frame, frame_center, (center_x, center_y), (0, 255, 255), 2)

        
    

    # Calculate FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    # Display FPS on screen
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break