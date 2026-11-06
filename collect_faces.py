import cv2
import os

# 初始化 Haar Cascade 人臉偵測器
cascade_path = 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"錯誤：無法載入模型檔 {cascade_path}")
    exit()

# 輸入使用者/學生 ID
student_id = input("請輸入人員/學生ID (例如: student01): ").strip()
save_dir = os.path.join('data', student_id)

# 建立資料夾
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
count = 0
max_samples = 50  # 預設擷取 50 張人臉樣本

print("[INFO] 開始收集人臉影像，請面向攝影機... 按 'q' 可提前退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("錯誤：無法取得攝影機影像")
        break

    # 轉為灰階影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 人臉偵測
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        count += 1
        # 裁切灰階人臉
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200)) # 統一解析度

        # 儲存影像
        file_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(file_path, face_roi)

        # 繪製標示框與計數文字
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {count}/{max_samples}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Collect Faces', frame)

    # 達到指定數量或按下 'q' 鍵退出
    if count >= max_samples or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] 影像收集完成，共儲存 {count} 張人臉至 {save_dir}")