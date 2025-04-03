import os
import cv2
import time
import numpy as np
from deepface import DeepFace
from deepface.DeepFace import build_model
from picamera2 import Picamera2

# === Tạo thư mục dataset ===
DATASET_FOLDER = "dataset"
os.makedirs(DATASET_FOLDER, exist_ok=True)

# === Nhập ID người dùng ===
user_id = input("Nhập ID cho người dùng: ")
user_folder = os.path.join(DATASET_FOLDER, user_id)
os.makedirs(user_folder, exist_ok=True)

# === Load mô hình ArcFace ===
arcface_model = build_model("ArcFace")

# === Khởi động camera ===
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (800, 600)}, controls={"FrameRate": 30})
picam2.configure(config)
picam2.start()
time.sleep(1)

# === Biến kiểm soát ===
count = 0
detecting_face = False

print(f"[INFO] Đang chụp ảnh cho {user_id}. Nhấn 'Q' để thoát...")

while count < 10:
    # Chụp ảnh từ camera
    frame = picam2.capture_array()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    faces = DeepFace.extract_faces(img_path=rgb_frame, enforce_detection=False)

    # Nếu có khuôn mặt
    if faces:
        if not detecting_face:
            print("✅ Phát hiện khuôn mặt! Bắt đầu chụp ảnh...")

        detecting_face = True  # Đánh dấu trạng thái nhận diện khuôn mặt

        # Chỉ lấy khuôn mặt lớn nhất
        largest_face = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
        facial_area = largest_face.get("facial_area", {})

        if all(k in facial_area for k in ("x", "y", "w", "h")):
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

            # Cắt ảnh khuôn mặt từ frame gốc
            face_img = frame[y:y+h, x:x+w]

            # Lưu ảnh chỉ chứa khuôn mặt
            img_filename = os.path.join(user_folder, f"{user_id}_{count}.jpg")
            cv2.imwrite(img_filename, face_img)
            print(f"[INFO] Đã lưu ảnh số {count+1}/10 tại: {img_filename}")
            count += 1

            # Hiển thị chỉ khuôn mặt trên cửa sổ (giúp dễ nhìn hơn)
            cv2.imshow("Khuôn mặt", face_img)

    else:
        if detecting_face:
            print("❌ Không phát hiện khuôn mặt! Dừng chụp ảnh...")
        detecting_face = False  # Đánh dấu không còn khuôn mặt

    # Hiển thị khung hình (chỉ có đường viền nhận diện)
    frame_display = frame.copy()
    if detecting_face:
        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Chụp Ảnh Khuôn Mặt", frame_display)

    # Dừng chương trình nếu nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Dọn dẹp ===
print("[INFO] Chụp ảnh hoàn tất! Đóng chương trình...")
cv2.destroyAllWindows()
picam2.stop()
picam2.close()
