
import cv2
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
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Camera (Grayscale)', gray_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space bar to capture an image
                image_path = os.path.join(save_dir, f'image_{len(images) + 1}.jpg')
                cv2.imwrite(image_path, gray_frame)
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
        loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if loaded_image is not None:
            loaded_images.append(loaded_image)
    return loaded_images

# Capture 10 images in grayscale and save them to the 'images' folder
image_paths = capture_images()

# Load captured grayscale images
images = load_images(image_paths)

# Display the loaded grayscale images
for i, img in enumerate(images):
    cv2.imshow(f'Image {i+1}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
