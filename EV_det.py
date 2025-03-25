'''
Hsiang 2025/03/25
優化影像前處理
新增中文辨識結合EV判斷機制
'''
import cv2
import numpy as np
import cv2
import torch
import numpy as np
import easyocr

model_path = "best.pt"      #yolo model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
reader = easyocr.Reader(['en'], gpu=True)  # 可加 'ch_tra' 來辨識繁體中文
reader_zh = easyocr.Reader(['ch_tra'], gpu=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
''''''
def rectify_plate(cropped_image):
    try:
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)#灰階
        
        blur = cv2.GaussianBlur(gray, (3, 3), 0)#高斯模糊
        edges = cv2.Canny(blur, 100, 200)#邊緣檢測
        
        #cv2.imshow("edges", edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#找輪廓
        # 繪製輪廓
        #contour_img = cropped_image.copy()
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(gray_bgr, contours, -1, (0, 255, 0), 1)
        cv2.imshow("Contours", gray_bgr) 

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                pts = approx.reshape(4, 2)

                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                rect = np.array([
                    pts[np.argmin(s)],
                    pts[np.argmin(diff)],
                    pts[np.argmax(s)],
                    pts[np.argmax(diff)]
                ], dtype='float32')

                (tl, tr, br, bl) = rect
                width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

                dst = np.array([[0, 0], [width - 1, 0],
                                [width - 1, height - 1], [0, height - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(cropped_image, M, (width, height))

                return warped
        return None
    except Exception as e:
        print(f"[錯誤] {e}")
        return None

# ========== 主測試程式 ==========
if __name__ == "__main__":
    warped = None
    cap = cv2.VideoCapture(2)  # 可改成影片路徑
    size = (1280, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    
    while True:
        is_ev = False
        is_ev_en = False
        is_ev_zh = False
        text = ""
        plate_text = ""
        text_color = (0, 255, 0) 
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        
        #YOLO
        results = model(frame)
        result = np.array(results.pandas().xyxy[0])

        for i in result:
            # 取得框選區域的座標
            x1, y1, x2, y2 = int(i[0]-20), int(i[1]-20), int(i[2]+20), int(i[3]+20)
            cropped = frame[y1:y2, x1:x2]#裁切區域

            if cropped.size > 0:  # 確保裁切區域不為空
                cv2.imshow("cropped", cropped)  # 顯示裁切區域
                
                # OCR(ZH)
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)#灰階
                ocr_results_zh = reader_zh.readtext(gray)
                for _, zh_text, conf in ocr_results_zh:
                    print(f"OCR 辨識結果：{zh_text}（信心 {conf:.2f}）")
                    if "電" in zh_text or "動" in zh_text or "車" in zh_text:
                        is_ev_zh = True
                    break  #只取第一筆
                
                # OCR(EN)                     測試過80效果不錯
                _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)#二值化 
                thresh = cv2.resize(thresh, (180, 80))    
                cv2.imshow("thresh", thresh)   
                ocr_results = reader.readtext(thresh)
                for _, text, conf in ocr_results:
                    text = text.replace(" ", "").upper()
                    plate_text = text.replace(" ", "").upper()
                    print(f"OCR 辨識結果：{text}（信心 {conf:.2f}）")
                    if text.startswith("E") or text.startswith("RE"):
                        is_ev_en = True
                    else:
                        text = "is not EV!"
                    break  #取第一筆
                
                print(is_ev_en, is_ev_zh)
                if is_ev_zh:
                    text = "Is_EV!"
                elif is_ev_en and is_ev_zh:
                    text = "Is_EV!"
                elif is_ev_en:    
                    text = "Is_EV! But Double Check!"
                else:
                    text = "is not EV!"
                    text_color = (0, 0, 255)
                cv2.putText(frame, text+"NUM:"+plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                '''
                #歪斜修正(效果極差待優化)
                #warped = rectify_plate(cropped)  # 拉平
                if warped is not None:  # 如果有拉平
                      # 顯示拉平後車牌
                    
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)#灰階
                    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)#二值化
                    cv2.imshow("thresh", thresh)  
                    ocr_results_zh = reader_zh.readtext(gray)
                    for _, zh_text, conf in ocr_results_zh:
                        print(f"OCR 辨識結果：{zh_text}（信心 {conf:.2f}）")
                        #text = "Is_EV!"
                        #cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break  # 只取第一筆'
                    
                    cv2.imshow("warped", warped)
                
                else:
                    print("未找到合適的四邊形輪廓")  # 未找到合適的四邊形輪廓
                '''

            else:
                print("Non_cropped")  # 裁切區域為空，跳過處理

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)#框選區域        
        cv2.imshow("DET", frame)#顯示原始圖+框選區域
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()