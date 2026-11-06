import cv2
import os
import json
import numpy as np

data_dir = 'data'
models_dir = 'models'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 初始化 LBPH 人臉辨識器
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_id = 0

print("[INFO] 開始讀取訓練資料...")

# 巡覽 data 目錄下的學生資料夾
for root, dirs, files in os.walk(data_dir):
    for dir_name in dirs:
        person_path = os.path.join(root, dir_name)
        if dir_name not in label_map:
            label_map[current_id] = dir_name
            target_id = current_id
            current_id += 1
        else:
            target_id = label_map[dir_name]

        for file in os.listdir(person_path):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_path, file)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if gray_img is None:
                    continue

                faces.append(gray_img)
                labels.append(target_id)

if len(faces) == 0:
    print("[ERROR] 未找到任何人臉訓練影像，請先執行 collect_faces.py收集資料！")
    exit()

print(f"[INFO] 載入 {len(faces)} 張影像，開始訓練 LBPH 模型...")
recognizer.train(faces, np.array(labels))

# 儲存模型與標籤
model_path = os.path.join(models_dir, 'lbph_model.yml')
labels_path = os.path.join(models_dir, 'labels.json')

recognizer.write(model_path)
with open(labels_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"[INFO] 訓練完成！模型已儲存至: {model_path}")
print(f"[INFO] 標籤對應表已儲存至: {labels_path}")