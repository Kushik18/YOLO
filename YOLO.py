import cv2
from ultralytics import YOLO

# Initialize YOLO model
yolo = YOLO('yolov8s.pt')

# Open the video capture (camera)
videoCap = cv2.VideoCapture(0)

# Check if the camera opened correctly
if not videoCap.isOpened():
    print("Error: Camera could not be accessed.")
else:
    print("Camera opened successfully.")

# Function to generate colors for bounding boxes
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
             (cls_num // len(base_colors)) % 5 for i in range(3)]
    return tuple(color)

# Start the video capture loop
while True:
    ret, frame = videoCap.read()
    
    # If frame capture fails, print an error message
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Print the dimensions of the frame for debugging
    print(f"Captured frame with shape: {frame.shape}")
    
    # Perform inference with YOLO
    results = yolo(frame)  # Use YOLO inference

    # Initialize counters for people and vehicles
    people_count = 0
    vehicle_count = 0

    # Iterate over detection results
    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:  # Confidence threshold for detection
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                class_name = classes_names[cls]

                # Count people and vehicles based on class
                if class_name == 'person':  # Count people
                    people_count += 1
                elif class_name in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:  # Count vehicles
                    vehicle_count += 1

                # Get color for bounding box
                colour = getColours(cls)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # Display the counts of people and vehicles on the frame
    cv2.putText(frame, f'People: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with detections and counts
    cv2.imshow('frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop...")
        break

# Release video capture and close windows
videoCap.release()
cv2.destroyAllWindows()