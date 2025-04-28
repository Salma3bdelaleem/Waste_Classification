import cv2
import time
import threading
from tempfile import NamedTemporaryFile
from gradio_client import Client, handle_file

client = Client("Salma3bdelaleem/Waste_Classification")

def send_to_api(frame):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, frame)
        try:
            result = client.predict(handle_file(tmp.name), api_name="/predict")
            print("Class:", result['label'])
        except Exception as e:
            print("Error:", e)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit...")

last_sent = 0
interval = 2  # seconds between API calls

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera", frame)

    if time.time() - last_sent > interval:
        threading.Thread(target=send_to_api, args=(frame.copy(),)).start()
        last_sent = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
