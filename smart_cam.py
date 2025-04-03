import os
import cv2
import time
import numpy as np
from datetime import datetime
from deepface import DeepFace
from deepface.DeepFace import build_model
from picamera2 import Picamera2
from threading import Thread, Lock
import traceback
import pickle
import hashlib

# === CẤU HÌNH ===
UNKNOWN_FOLDER = "unknown_faces"
CAPTURE_INTERVAL = 0.5  # giây
THRESHOLD_PROBA = 0.9  # Tăng độ tin cậy để nhận người quen chính xác hơn

# === KHỞI TẠO ===
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
frame_lock = Lock()
latest_frame = None
running = True
last_capture_time = {}

# === TẢI MODEL ARC FACE ===
try:
    model = build_model("ArcFace")
except Exception as e:
    print(f"❌ Lỗi khi tải model ArcFace: {e}")
    exit()

# === TẢI MÔ HÌNH PHÂN LOẠI clf.pkl ===
try:
    with open("clf.pkl", "rb") as f:
        clf = pickle.load(f)
    print("✅ Đã tải mô hình phân loại (clf.pkl)")
except Exception as e:
    print(f"❌ Lỗi khi tải clf.pkl: {e}")
    exit()

# === CAMERA CONFIG ===
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    controls={
        "FrameRate": 20,
        "AnalogueGain": 2.0,         # tăng sáng
        "ExposureTime": 15000,       # thời gian phơi sáng (microseconds)
        "AwbEnable": True            # bật auto white balance
        }
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(1)
except Exception as e:
    print(f"❌ Lỗi khi khởi động camera: {e}")
    exit()

# === CẢI THIỆN ẢNH ===
def enhance_image(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        sharpened = cv2.filter2D(enhanced, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return sharpened
    except Exception as e:
        print("❌ Lỗi enhance ảnh:", e)
        return frame

# === THREAD CAMERA ===
def camera_thread():
    global latest_frame, running
    while running:
        try:
            frame = picam2.capture_array()
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.03)
        except:
            print("❌ Lỗi camera:", traceback.format_exc())

class StrangerTracker:
    def __init__(self):
        self.tracks = {}  # key = unique_face_id
        self.lock = Lock()

    def update(self, face_id, bbox, timestamp):
        with self.lock:
            if face_id not in self.tracks:
                self.tracks[face_id] = {
                    "start": timestamp,
                    "last": timestamp,
                    "bbox": bbox
                }
            else:
                self.tracks[face_id]["last"] = timestamp
                self.tracks[face_id]["bbox"] = bbox

    def get_strangers_standing_too_long(self, timeout=10):
        warnings = []
        now = time.time()
        with self.lock:
            for face_id, info in self.tracks.items():
                duration = info["last"] - info["start"]
                if duration > timeout:
                    warnings.append((face_id, duration, info["bbox"]))
        return warnings
def behavior_thread():
    global running
    while running:
        strangers = tracker.get_strangers_standing_too_long()
        for face_id, duration, bbox in strangers:
            print(f"[⚠️ CẢNH BÁO] Người lạ ID {face_id} đứng quá lâu ({duration:.1f}s) tại {bbox}")
            # TODO: Gửi cảnh báo, ghi log, hoặc kích hoạt còi
        time.sleep(1)

# === THREAD NHẬN DIỆN ===
def recognition_thread():
    global latest_frame, running
    last_detect_time = 0
    DETECT_INTERVAL = 0.5  # Giảm tần suất để tăng hiệu suất

    while running:
        try:
            now = time.time()
            if now - last_detect_time < DETECT_INTERVAL:
                time.sleep(0.01)
                continue
            last_detect_time = now

            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None

            if frame is None:
                continue

            processed = enhance_image(frame)
            small_frame = cv2.resize(processed, (0, 0), fx=0.4, fy=0.4)  # resize nhỏ hơn để tăng tốc
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            try:
                faces = DeepFace.extract_faces(img_path=rgb_frame, enforce_detection=True)
            except:
                faces = []

            if not faces:
                continue

            output_frame = processed.copy()

            for idx, face in enumerate(faces):
                area = face.get("facial_area", {})
                if not all(k in area for k in ("x", "y", "w", "h")):
                    continue

                scale = 2.5  # scale tương ứng lại do ảnh nhỏ hơn
                x = int(area["x"] * scale)
                y = int(area["y"] * scale)
                w = int(area["w"] * scale)
                h = int(area["h"] * scale)

                if x < 0 or y < 0 or x + w > output_frame.shape[1] or y + h > output_frame.shape[0]:
                    continue
                if w * h < 5000:
                    continue

                face_img = processed[y:y + h, x:x + w]
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < 60 or np.std(gray) < 8:
                    continue

                try:
                    emb_obj = DeepFace.represent(img_path=face["face"], model_name="ArcFace", enforce_detection=False)
                    if not emb_obj or "embedding" not in emb_obj[0]:
                        continue
                    emb = emb_obj[0]["embedding"]
                except:
                    continue

                try:
                    probas = clf.predict_proba([emb])[0]
                    pred_idx = np.argmax(probas)
                    pred_label = clf.classes_[pred_idx]
                    proba = probas[pred_idx]

                    if pred_label == "not_face" and proba > 0.9:
                        name = "Không phải mặt người"
                    elif pred_label != "not_face" and proba > THRESHOLD_PROBA:
                        name = pred_label
                    else:
                        name = "Người lạ"
                except:
                    name = "Không nhận diện"
                    proba = 0.0

                print(f"[DEBUG] Mặt {idx+1}: {name} | Tin cậy: {proba:.2f}")

                # Màu khung
                if name == "Người lạ":
                    color = (0, 0, 255)
                elif name == "Không phải mặt người":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                # Vẽ khung và nhãn
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(output_frame, f"{name} ({proba:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # === NGƯỜI LẠ ===
                if name == "Người lạ":
                    current_time = time.time()

                    emb_str = ",".join([f"{x:.4f}" for x in emb])
                    unique_id = hashlib.md5(emb_str.encode()).hexdigest()

                    tracker.update(unique_id, (x, y, w, h), current_time)

                    if unique_id not in last_capture_time or (current_time - last_capture_time[unique_id]) > CAPTURE_INTERVAL:
                        last_capture_time[unique_id] = current_time
                        filename = f"{UNKNOWN_FOLDER}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id[:6]}.jpg"
                        cv2.imwrite(filename, face_img)
                        print(f"[ALERT] Lưu ảnh người lạ: {filename}")

            # Thời gian
            now_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(output_frame, f"Thoi gian: {now_time}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with frame_lock:
                latest_frame = output_frame.copy()

        except Exception as e:
            print("❌ Lỗi nhận diện:", traceback.format_exc())

        time.sleep(0.05)
# === CHẠY CHƯƠNG TRÌNH ===
tracker = StrangerTracker()  # Khởi tạo tracker người lạ

Thread(target=camera_thread, daemon=True).start()
Thread(target=recognition_thread, daemon=True).start()
Thread(target=behavior_thread, daemon=True).start()  # Thêm thread hành vi


cv2.namedWindow("Camera AI - Nhận diện khuôn mặt", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera AI - Nhận diện khuôn mặt", 1280, 720)

try:
    while running:
        with frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None
        if frame_copy is not None:
            cv2.imshow("Camera AI - Nhận diện khuôn mặt", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
except KeyboardInterrupt:
    running = False
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("[INFO] Thoát chương trình.")
