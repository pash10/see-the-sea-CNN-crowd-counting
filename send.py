import time
import threading
import queue
import cv2
import numpy as np 
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db,storage
from count import process_frame  # Import functions from count.py
from flask import Flask
import matplotlib.pyplot as plt64
import base64

# Initialize Flask app
app = Flask(__name__)


# Initialize Firebase Admin SDK
cred = 
firebase_admin.initialize_app(cred, {

# Global variables
video_stream_url = ''
frame_queue = queue.Queue(maxsize=60)  # Queue to store frames
count = 0
hold_img_numer = queue.Queue(maxsize=60)


def fetch_ip():
    global video_stream_url
    try:
        ref = db.reference('/esp32cam/ip')
        url_db2 = ref.get()
        if not url_db2:
            print("No IP address found")
            return

        print("Fetched ESP32 IP Address:", url_db2)
        video_stream_url = url_db2
    except Exception as e:
        print(f"Error fetching IP: {e}")



def producer():
    global count  # Access the global count variable

    cap = cv2.VideoCapture(video_stream_url)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)

            # Safely increment the count
            with threading.Lock():
                count += 1
                hold_img_numer.put(count)
                print(f"pic num {count}")

            # Upload the frame to Firebase Storage
            upload_frame_to_firebase(frame, count)

        time.sleep(0.5)  # Adjust based on your requirements


def upload_frame_to_firebase(frame, frame_id):
    # Convert the frame to JPEG format
    _, buffer = cv2.imencode('.jpg', frame)
    img_bytes = buffer.tobytes()

    # Encode the image to base64 string
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Get a reference to the Firebase Realtime Database
    ref = db.reference(f'frames/{frame_id}')

    # Set the base64 string to the database
    ref.set({
        'image': img_base64
    })
    print(f"Uploaded frame {frame_id} to Firebase Realtime Database.")




def consumer():
    ref = db.reference('/count')
    while True:
        if not hold_img_numer.empty() and not frame_queue.empty():
            frame = frame_queue.get()
            print("Frame Type:", type(frame))
            print("Frame Shape:", frame.shape if isinstance(frame, np.ndarray) else "Not a NumPy array")

            # Let's assume process_frame now returns both count and the image to be plotted
            count, img_to_plot = process_frame(frame)
            img_number = hold_img_numer.get()
            plot_img(img_number, count, img_to_plot)  # Pass the image for plotting
            update_count(ref, count)




def plot_img(img_number, count, img):
    # Ensure the image is in 8-bit format without changing its size
    if img.dtype != np.uint8:
        img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_8bit = np.uint8(img_normalized)
    else:
        img_8bit = img

    # Ensure grayscale for colormap application without resizing
    if len(img_8bit.shape) == 3 and img_8bit.shape[2] == 3:
        img_gray = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_8bit

    # Apply a colormap directly on the original-sized grayscale image
    heatmap_img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)

    # Overlay text directly on the heatmap without altering its size
    cv2.putText(heatmap_img, f"Num of people: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(heatmap_img, f"Pic num: {img_number}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the heatmap while keeping the original image size intact
    window_name = f"Heatmap {img_number}"
    cv2.imshow(window_name, heatmap_img)
    cv2.waitKey(2000)  # Display for 2 seconds
    cv2.destroyWindow(window_name)


def update_count(ref, count):
    if count is not None:
        ref.set(int(count))


def convert_to_pil_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return pil_image

def convert_to_numpy_array(pil_image):
    frame = np.array(pil_image)
    return frame

def generate_video_stream():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, frame_bytes = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error encoding frame")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)  # Add a slight delay to avoid busy-waiting

#need to test 
#@app.route('/video_feed')  
#def video_feed():
#   return Response(generate_video_stream(),
#                   mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask_app():
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=8080, threaded=True)

# Main Execution
if __name__ == '__main__':
    print("Starting main execution...")
    fetch_ip()
    
    flask_thread = threading.Thread(target=run_flask_app)
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    flask_thread.start()
    producer_thread.start()
    consumer_thread.start()

    # Join threads
    flask_thread.join()
    producer_thread.join()
    consumer_thread.join()

