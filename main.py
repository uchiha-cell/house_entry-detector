import cv2
import torch
import smtplib
from email.message import EmailMessage
from config import *
import os
from datetime import datetime

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load your input video
cap = cv2.VideoCapture('entry_video.mp4')

if not cap.isOpened():
    print("Error opening video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_size = (frame_width, frame_height)

out = cv2.VideoWriter('output_with_boxes.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      frame_size)

# Folder to save detected person images permanently
os.makedirs('detected_persons', exist_ok=True)

notified = False

def send_email(message, image_path=None):
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Alert: Person Detected in Video'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content(message)

        # Attach image if provided
        if image_path:
            with open(image_path, 'rb') as img:
                img_data = img.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("✅ Email sent with photo." if image_path else "✅ Email sent.")
    except Exception as e:
        print("❌ Failed to send email:", e)

def save_person_image(image, count):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'detected_persons/person_{timestamp}_{count}.jpg'
    cv2.imwrite(filename, image)
    return filename

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.pred[0]

    person_count = 0
    image_paths = []

    for (*box, conf, cls) in detections:
        class_name = model.names[int(cls)]
        if class_name == 'person' and conf > 0.5:
            person_count += 1
            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {person_count}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Crop and save the detected person image
            person_img = frame[y1:y2, x1:x2]
            saved_path = save_person_image(person_img, person_count)
            image_paths.append(saved_path)

    out.write(frame)
    cv2.imshow("Detection", frame)

    # Send email only once when person(s) detected
    if person_count > 0 and not notified:
        # Send email with message and first person's photo (you can extend to send all)
        send_email(f"{person_count} person(s) detected in the video.", image_paths[0])
        notified = True

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Finished processing video.")
