from ultralytics import YOLO

model = YOLO('./runs/detect/complete_model/weights/best.pt')


# inference
results = model.predict("input/08fd33_4.mp4", save=True)
print(results[0])
print("==============================")
for box in results[0].boxes:
    print(box)