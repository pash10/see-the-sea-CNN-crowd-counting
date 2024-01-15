import time
import threading
import queue
import cv2
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from count import load_model, create_img, predict  # Import functions from count.py

# Initialize Firebase Admin SDK
cred = credentials.Certificate('see-the-sea-4c396-firebase-adminsdk-q002e-b260d83c4c.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://see-the-sea-4c396-default-rtdb.europe-west1.firebasedatabase.app'})

# Global variables
video_stream_url = ''
frame_queue = queue.Queue(maxsize=10)  # Queue to store frames

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
    cap = cv2.VideoCapture(video_stream_url)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)
        time.sleep(0.5)

def consumer(model):
    ref = db.reference('/count')
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            count, _, _ = predict(model, pil_image)
            update_count(ref, count)

def update_count(ref, count):
    if count is not None:
        ref.set(count)

def main():
    fetch_ip()
    model = load_model()
    if model is None:
        print("Error loading model")
        return

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer, args=(model,))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

if __name__ == "__main__":
    main()
