import cv2
import json
import os

cascade_path = 'haarcascade_frontalface_alt.xml'
model_path = 'models/lbph_model.yml'
labels_path = 'models/labels.json'

# 檢查檔案是否存在
if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print("[ERROR] 找不到模型或標籤檔，請先執行 train_lbph.py 完成訓練！")
    exit()

# 載入 Haar Cascade 與 LBPH 模型
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# 載入類別標籤
with open(labels_path, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
    # json key 預設為 string，轉回 int
    label_map = {int(k): v for k, v in label_map.items()}

# 開啟 Webcam
cap = cv2.VideoCapture(0)
threshold = 80.0  # LBPH 距離閾值 (數值越小相似度越高，超過此值判定為 Unknown)

print("[INFO] 啟動即時辨識系統... 按 'q' 鍵退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        # LBPH 特徵分析與比對
        label_id, distance = recognizer.predict(face_roi)

        # 距離判斷 (距離 <= 閾值 ?)
        if distance <= threshold:
            name = label_map.get(label_id, "Unknown")
            confidence_text = f"{name} ({int(distance)})"
            color = (0, 255, 0) # 綠色
        else:
            name = "Unknown"
            confidence_text = f"Unknown ({int(distance)})"
            color = (0, 0, 255) # 紅色

        # 繪製標示與名稱
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, confidence_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('LBPH Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()