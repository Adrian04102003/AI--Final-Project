import cv2
import time
from pathlib import Path

# Config
DATA_DIR = Path('./data')
NUMBER_OF_CLASSES = 38
DATASET_SIZE_PER_CLASS = 100
WEBCAM_INDEX = 0

DATA_DIR.mkdir(exist_ok=True) 

cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
    exit()

print(f"Starting data collection for {NUMBER_OF_CLASSES} classes...")


for class_index in range(NUMBER_OF_CLASSES):
    
    class_dir = DATA_DIR / str(class_index)
    class_dir.mkdir(exist_ok=True)
    
    print(f"\n--- Collecting data for Class {class_index}/{NUMBER_OF_CLASSES-1} ---")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
            
        text = f'Ready for Class {class_index}? Press "Q" to START!'
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collector', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    image_counter = 0
    start_time = time.time()
    
    while image_counter < DATASET_SIZE_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
            
        progress_text = f'Class: {class_index} | Collecting: {image_counter}/{DATASET_SIZE_PER_CLASS}'
        cv2.putText(frame, progress_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Data Collector', frame)
        
        image_path = class_dir / f'{image_counter}.jpg'
        cv2.imwrite(str(image_path), frame)
        
        cv2.waitKey(25) 
        
        image_counter += 1

    print(f"Collected {image_counter} images for Class {class_index}.")
    
# Clean
cap.release()
cv2.destroyAllWindows()
print("\nData collection finished.")