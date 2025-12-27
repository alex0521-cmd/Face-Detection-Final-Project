import numpy as np
import argparse
import cv2
import os


# --- 支援中文路徑的讀取函式 ---
def cv2_imread(file_path):
    # DNN 建議使用 IMREAD_COLOR (3通道)
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


# --- 參數設定 ---
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="輸入照片資料夾")
args = vars(ap.parse_args())

# 1. 取得模型路徑
base_dir = os.path.dirname(os.path.abspath(__file__))
proto_path = os.path.join(base_dir, "deploy.prototxt")
model_path = os.path.join(base_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# 2. 載入 DNN 模型
if not os.path.exists(proto_path) or not os.path.exists(model_path):
    print(f"[錯誤] 找不到模型檔案，請確認檔案在：{base_dir}")
    exit()

print("[INFO] 正在載入 DNN 模型...")
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# 3. 遍歷資料夾進行偵測
for filename in os.listdir(args["folder"]):
    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
        continue

    full_path = os.path.join(args["folder"], filename)
    img = cv2_imread(full_path)
    if img is None: continue

    (h, w) = img.shape[:2]

    # 4. DNN 影像預處理 (縮放至 300x300)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # 5. 繪製偵測結果
    face_count = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 信心度大於 50% 才顯示
        if confidence > 0.5:
            face_count += 1
            # 計算臉部座標
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 確保座標不超出圖片邊界
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # 畫出綠色方框與信心度
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"Face: {confidence * 100:.1f}%"
            cv2.putText(img, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"[處理成功] {filename}: 發現 {face_count} 張臉")

    # 6. 縮放顯示結果 (避免圖片太大超出螢幕)
    display_w = 800
    if w > display_w:
        display_h = int(h * (display_w / w))
        img_display = cv2.resize(img, (display_w, display_h))
    else:
        img_display = img

    cv2.imshow("DNN Face Detection (Press any key for next, Q to quit)", img_display)

    # 等待按鍵：按 'q' 退出，按其他鍵繼續下一張
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("[INFO] 程式結束")