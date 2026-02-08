import os
import cv2
from ultralytics import YOLO

def count_objects_in_dataset(dataset_dir, model_path):
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return None

    total_detections = 0
    all_detections_per_frame = {}
    image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print("No image or video files found in the dataset directory.")
        return None

    for file_name in image_files:
        file_path = os.path.join(dataset_dir, file_name)
        results = model(file_path, stream=True)
        frame_detections = 0
        
        for r in results:
            detections_count = len(r.boxes)
            frame_detections += detections_count
            total_detections += detections_count
        
        all_detections_per_frame[file_name] = frame_detections
        print(f"Count for frame '{file_name}': {frame_detections}")

    print("\n" + "-"*30)
    print("Detection Summary:")
    print(f"Total objects detected across all frames: {total_detections}")
    print("\nDetections per frame (First 5 images):")
    
    for file, count in list(all_detections_per_frame.items())[:5]:
        print(f"  {file}: {count}")

    return all_detections_per_frame


dataset_directory = '/home/javra/work/yolo/object_detection_frame/new_data/Data/images'
model_file_path = '/home/javra/work/yolo/object_detection_frame/runs/detect/train73/weights/best.pt'

count_objects_in_dataset(dataset_directory, model_file_path)

for file, count in list(all_detections_per_frame.items())[:5]:
    print(f" {file}: {count}")
    
    file_path = os.path.join(dataset_dir, file)
    results = model(file_path) 
    
    for r in results:
        annotated_frame = r.plot()  
        cv2.imshow(f"{file} - Count: {count}", annotated_frame)
        cv2.waitKey(0)  
    
cv2.destroyAllWindows()
