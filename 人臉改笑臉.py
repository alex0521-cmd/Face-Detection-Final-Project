import numpy as np
import argparse
import cv2
import os


# --- 支援中文路徑的讀取與存檔 ---
def cv2_imread(file_path, flags=cv2.IMREAD_UNCHANGED):
    # 注意：這裡加上了 IMREAD_UNCHANGED，確保能讀取到第 4 個 Alpha 通道
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
    return cv_img


def cv2_imwrite(file_path, img):
    ext = os.path.splitext(file_path)[1]
    result, nparr = cv2.imencode(ext, img)
    if result:
        nparr.tofile(file_path)
        return True
    return False


# --- 參數設定 ---
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-s", "--sticker", required=True)  # 建議使用帶透明背景的 PNG
args = vars(ap.parse_args())

input_folder = args["folder"]
output_folder = args["output"]
sticker_path = args["sticker"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. 讀取貼圖 (包含 Alpha 通道)
sticker = cv2_imread(sticker_path, cv2.IMREAD_UNCHANGED)
if sticker is None or sticker.shape[2] != 4:
    print("錯誤：請確保貼圖是具有透明背景的 PNG 檔案！")
    exit()

faceCascade = cv2.CascadeClassifier('face_dectet.xml')

for filename in os.listdir(input_folder):
    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
        continue

    full_input_path = os.path.join(input_folder, filename)
    img = cv2_imread(full_input_path, cv2.IMREAD_COLOR)
    if img is None: continue

    # 縮放主圖
    (h, w) = img.shape[:2]
    target_width = 720
    if w > target_width:
        ratio = target_width / float(w)
        img = cv2.resize(img, (target_width, int(h * ratio)))

    # 偵測人臉
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    faceRect = faceCascade.detectMultiScale(gray, 1.1, 5)

    # 2. 處理每一張臉的透明合成
    for (x, y, w_box, h_box) in faceRect:
        # 縮放貼圖以符合人臉大小
        s_resized = cv2.resize(sticker, (w_box, h_box))

        # 分離貼圖的 BGR 顏色與 Alpha 遮罩
        s_bgr = s_resized[:, :, 0:3]  # 貼圖顏色 (前 3 通道)
        alpha = s_resized[:, :, 3] / 255.0  # 透明度 (第 4 通道)，轉為 0.0~1.0

        # 取得原圖的人臉區域 (背景)
        bg_roi = img[y:y + h_box, x:x + w_box]

        # --- Alpha Blending 公式 ---
        # 結果 = 貼圖 * Alpha + 背景 * (1 - Alpha)
        # 這裡需要對三個顏色通道分別計算，所以我們用 np.newaxis 讓 alpha 維度一致
        alpha = alpha[:, :, np.newaxis]

        composite = (s_bgr * alpha + bg_roi * (1 - alpha)).astype(np.uint8)

        # 將合成後的結果貼回原圖
        img[y:y + h_box, x:x + w_box] = composite

    # 3. 存檔與顯示
    full_output_path = os.path.join(output_folder, f"alpha_{filename}")
    cv2_imwrite(full_output_path, img)

    cv2.imshow('Alpha Blending Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("處理完成！")