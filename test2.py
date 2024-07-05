import cv2
import face_recognition
import numpy as np
import os

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_images(num_images=10, save_dir='images'):
    ensure_directory(save_dir)
    camera = cv2.VideoCapture(0)
    images = []
    
    print("Press the space bar to capture an image. Press 'q' to quit.")
    
    while len(images) < num_images:
        ret, frame = camera.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Camera', rgb_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar to capture an image
                image_path = os.path.join(save_dir, f'image_{len(images) + 1}.jpg')
                cv2.imwrite(image_path, frame)  # Save original BGR image
                images.append(image_path)
                print(f'Captured image {len(images)}')
            elif key == ord('q'):  # 'q' to quit
                break
    
    camera.release()
    cv2.destroyAllWindows()
    return images

def load_images(images):
    loaded_images = []
    for image_path in images:
        loaded_image = cv2.imread(image_path)
        if loaded_image is not None:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
            loaded_images.append(rgb_image)
    return loaded_images

def live_face_recognition(known_face_path):
    # Load the known face image and encode it
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize webcam
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare each face found with the known face
            results = face_recognition.compare_faces([known_face_encoding], face_encoding)
            name = "Unknown"

            if results[0]:
                
                name = "Known Person"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

# Path to the known face image
known_face_path = 'D:\Face Recognition\Face-Reg\aki.jpg'  # Update with your known face image path

# Perform live face recognition
live_face_recognition(known_face_path)
