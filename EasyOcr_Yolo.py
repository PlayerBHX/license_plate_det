import torch
import cv2
import numpy as np
import time
import easyocr

# 初始化 EasyOCR
reader = easyocr.Reader(['en'])  # 可加 'ch_tra' 來辨識繁體中文

# Model
model_path = r"best.pt"  # Custom model path
video_path = r"anpr_video.mp4"  # Input video path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = model.to(device)

# Video Capture
frame = cv2.VideoCapture(2)#改錄影裝置
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
                #print(f"OCR 辨識結果: {text}")

                # 處理文字（去空格、轉大寫）
                plate_text = text.replace(" ", "").upper()

                # 預設為一般車
                is_ev = False
                text_color = (0, 0, 255)  

                # 如果是以 E 開頭，判斷為電動車
                if plate_text.startswith("E") or plate_text.startswith("RE"):
                    is_ev = True
                    text=">> DET EV"#Electric Vehicle
                    text_color = (0, 255, 0)  
                    
                    print(">> 偵測到電動車！")
                else:
                    text = "is not EV"
                # 顯示文字
                cv2.putText(image, f"{text}", text_origin, text_font, 1, text_color, 2)

                # 顯示結果
                #cv2.imshow("cropped_region", cropped_region)
                #cv2.imshow("thresh", thresh)
            
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
