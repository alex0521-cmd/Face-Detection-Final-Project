import numpy as np
import argparse
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="Path to the folder containing images")
args = vars(ap.parse_args())

folder_path = args["folder"]


for filename in os.listdir(folder_path):

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue

    full_file_path = os.path.join(folder_path, filename)

    print(f"正在處理: {filename} ...")

    img = cv2.imread(full_file_path)
    if img is None:
        print(f"無法讀取 {filename}，跳過。")
        continue

    (h, w) = img.shape[:2]
    target_width = 720

    if w > target_width:
        ratio = target_width / float(w)
        new_height = int(h * ratio)
        img = cv2.resize(img, (target_width, new_height))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    faceCascade = cv2.CascadeClassifier('face_dectet.xml')


    faceRect = faceCascade.detectMultiScale(blurred, 1.1, 5)

    print(f" -> 偵測到 {len(faceRect)} 張臉")

    for (x, y, w, h) in faceRect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Batch Processing', img)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        print("使用者中斷程式")
        break

cv2.destroyAllWindows()