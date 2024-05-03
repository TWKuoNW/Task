import cv2
import torch
import numpy as np
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
cap = cv2.VideoCapture('D:/video/t2.MOV')

points = np.array([[0, 450], [640, 390], [640, 480], [0, 480]])

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model([img], size=640)

    results = results.xyxy[0].numpy()
    print(results)
    person_count = 0

    # cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    for *xyxy, conf, cls_id in results:
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    frame_count = frame_count + 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_fps = frame_count / elapsed_time

    cv2.putText(frame, f'People: {person_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'FPS: {actual_fps:.2f}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
