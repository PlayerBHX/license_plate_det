import torch
import cv2
import numpy as np
import time
import easyocr

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 初始化 EasyOCR
reader = easyocr.Reader(['en'])  # 可加 'ch_tra' 來辨識繁體中文

# Model
model_path = r"best.pt"  # Custom model path
video_path = r"anpr_video.mp4"  # Input video path
cpu_or_cuda = "cpu"  # Choose "cpu" or "cuda"
device = torch.device(cpu_or_cuda)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = model.to(device)

# Video Capture
frame = cv2.VideoCapture(2)
frame_width = int(frame.get(3))
frame_height = int(frame.get(4))
size = (frame_width, frame_height)

# Parameters
text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25
prev_frame_time = 0
new_frame_time = 0

# Inference Loop
while True:
    ret, image = frame.read()
    if ret:
        output = model(image)
        result = np.array(output.pandas().xyxy[0])

        for i in result:
            # 取得框選區域的座標
            x1, y1, x2, y2 = int(i[0]), int(i[1]), int(i[2]), int(i[3])
            text_origin = (x1, y1 - 5)

            # 繪製偵測框
            cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(image, text=f"{i[-1]} {i[-3]:.2f}", org=text_origin,
                        fontFace=text_font, fontScale=text_font_scale,
                        color=color, thickness=2)

            # 擷取車牌區域
            cropped_region = image[y1:y2, x1:x2]
            
            # 確保裁剪區域不為空
            if cropped_region.size == 0:
                continue

            # 轉為灰階
            gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

            # 濾波去雜訊
            gray = cv2.medianBlur(gray, 3)

            # OTSU 二值化
            _, thresh = cv2.threshold(gray, 0, 155, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 使用 EasyOCR 進行辨識
            ocr_results = reader.readtext(thresh)
            
            for bbox, text, prob in ocr_results:
                print(f"OCR 辨識結果: {text} (信心: {prob:.2f})")
                cv2.putText(cropped_region, text, (17, 17), text_font, 1, (0, 255, 0), 2)
                cv2.imshow("cropped_region", cropped_region)
                cv2.imshow("thresh", thresh)
            '''
            text = pytesseract.image_to_string(thresh, config=("--psm 8"))
            print(f"OCR 辨識結果: {text}")
            cv2.putText(cropped_region, text, (27, 27), text_font, 1, (0, 255, 0), 2)
            cv2.imshow("cropped_region", cropped_region)
            '''
            
        # 計算 FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # 顯示 FPS
        cv2.putText(image, f"FPS: {fps}", (10, 50), text_font, 1, (100, 255, 0), 2)

        # 顯示影像
        cv2.imshow("Result", image)

    else:
        break

    # 按 `q` 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()
