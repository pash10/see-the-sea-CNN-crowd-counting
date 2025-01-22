import time
import threading
import queue
import cv2
import numpy as np 
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
from count import process_frame  # Import functions from count.py
import base64
import atexit
import os

class ImgAndNum:
     def __init__(self,count,frame):
         self.count = count
         self.frame = frame
# Initialize Firebase Admin SDK
cred = credentials.Certificate('see-the-sea-4c396-firebase-adminsdk-q002e-5a3e258116.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://see-the-sea-4c396-default-rtdb.europe-west1.firebasedatabase.app'})

# Global variables
video_stream_url = ''
frame_queue = queue.Queue()  # Queue to store frames
count = 0
#hold_img_numer = queue.Queue()
to_con_q = queue.Queue()

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


def monitor_queue_empty_state(queue, empty_wait_time=30):
    while True:
        if queue.empty():
            print("Queue is empty. Starting timer.")
            start_time = time.time()

            while queue.empty():
                time.sleep(1)  # Sleep to prevent a tight loop
                if time.time() - start_time >= empty_wait_time:
                    print("Queue has been empty for 30 seconds. Triggering cleanup.")
                    cleanup()
                    return  # Exit the monitoring function
        else:
            time.sleep(1)  # Sleep briefly to wait for queue to potentially empty


def producer(video_stream_url):
    global count

    cap = cv2.VideoCapture(video_stream_url)
    while True:
        ret, frame = cap.read()
        if ret:
            temp= ImgAndNum(count,frame)
            frame_queue.put(temp)
            with threading.Lock():
                count += 1
                if count % 5 == 0: # if wifi slow change
                    to_con_q.put(temp)
                upload_frame_to_firebase()
                #hold_img_numer.put(count)
            print(f"Frame {count} added to the queue.")

            # Upload the frame to Firebase Storage

        time.sleep(0.5)  # Adjust as needed

def upload_frame_to_firebase():
    temp = frame_queue.get()
    # Convert the frame to JPEG format
    _, buffer = cv2.imencode('.jpg', temp.frame)
    img_bytes = buffer.tobytes()

    # Encode the image to base64 string
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Get a reference to the Firebase Realtime Database
    ref = db.reference(f'frames/{1}')

    # Set the base64 string to the database
    ref.set({
        'image': img_base64
    })
    print(f"Uploaded frame {temp.count} to Firebase Realtime Database.")

def consumer():
    global count
    ref = db.reference('/count')
    while True:
        if not to_con_q.empty():
            img = to_con_q.get()
            print("if work")
            print("Frame Type:", type(img.frame))
            print("Frame Shape:", img.frame.shape if isinstance(img.frame, np.ndarray) else "Not a NumPy array")

            # Let's assume process_frame now returns both count and the image to be plotted
            num, img_to_plot = process_frame(img.frame)
            plot_img(img.count, count, img_to_plot)  # Pass the image for plotting
            update_count(ref, num)
    else:
            # The condition was not met, print the states of the queues and the count
            print("Condition not met for processing.")
            #print(f"hold_img_numer empty: {hold_img_numer.empty()}, frame_queue empty: {frame_queue.empty()}, count: {count}")
            time.sleep(1)  # Sleep to prevent a tight loop and give time f                

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

# Define your cleanup function
def cleanup():
    # Get a reference to the Firebase Realtime Database
    ref = db.reference('/')
    # Set the count to 0
    ref.child('count').set(0)
    # Clear the frame data
    ref.child('frames/1/image').set('')
    print("Cleanup completed, set count to 0 and cleared frame data.")


def check_for_stop_signal():
    # Path to the stop signal file
    stop_signal_path = 'stop_signal.txt'
    return os.path.exists(stop_signal_path)

atexit.register(cleanup)

# Main Execution
if __name__ == '__main__':
    try:
        print("Starting main execution...")
        fetch_ip()
        
 # Initialize threads
        producer_thread = threading.Thread(target=producer, args=(video_stream_url,))
        consumer_thread = threading.Thread(target=consumer)
        queue_monitor_thread = threading.Thread(target=monitor_queue_empty_state, args=(frame_queue,))

        # Start threads
        producer_thread.start()
        consumer_thread.start()
        queue_monitor_thread.start()

        print("All threads started. Main execution continues...")

        # Wait for threads to complete
        producer_thread.join()
        consumer_thread.join()
        queue_monitor_thread.join()

        print("Interrupt received, cleaning up...")
        cleanup()
        print("See ya mate <3")
    except KeyboardInterrupt:
        print("Interrupt received, cleaning up...")
        cleanup()

