import cv2  
import torch 
import numpy as np  
import time  

def determine_crossing(A, B, C, D):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    AB = B - A
    CD = D - C

    def solve_linear_equation(AB, CD, C, A):
        matrix = np.array([AB, -CD]).T
        try:
            return np.linalg.solve(matrix, C - A)
        except np.linalg.LinAlgError:
            return None 

    result = solve_linear_equation(AB, CD, C, A)
    if result is not None:
        t, u = result
        if 0 <= t <= 1 and 0 <= u <= 1:
            if t > 0:
                return 1 if AB[0] * CD[1] - AB[1] * CD[0] > 0 else -1
            else:
                return -1 if AB[0] * CD[1] - AB[1] * CD[0] > 0 else 1
        else:
            return 0 
    else:
        return 0 

# 加載 yolov5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # 將模型設置為評估模式
cap = cv2.VideoCapture('D:/video/t2.MOV')

trajectories = []  # 初始化軌跡列表
person_in_count = 0  # 初始化進入區域的人數計數器
person_out_count = 0  # 初始化離開區域的人數計數器

# 定義一個區域的四個點
p1 = [0, 625] # 左上
p2 = [1366, 600] # 右上
p3 = [1366, 768] # 右下
p4 = [0, 768] # 左下

points = np.array([p1, p2, p3, p4], dtype=np.int32) 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/video/output2.avi', fourcc, 20.0, (1366, 768))

frame_count = 0  # 初始化fps計數器
start_time = time.time()  # 記錄開始處理時間

while(cap.isOpened()):
    ret, frame = cap.read()  # 讀取一幀
    if not ret:
        break  # 如果無法讀取，跳出循環

    frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)  # 調整畫面的大小
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 將幀轉換成 RGB
    results = model([img], size=640)  # 使用模型進行預測
    results = results.xyxy[0].numpy()  # 提取結果並轉換為 NumPy 陣列

    cv2.polylines(frame, [points], isClosed=True, color=(255, 100, 100), thickness=2)  # 繪製多邊形

    overlay = np.zeros_like(frame)
    cv2.fillPoly(overlay, [points], color=(255, 100, 100))
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


    new_trajectories = []  # 初始化新的軌跡列表
    for *xyxy, conf, cls_id in results:  # 迭代檢測到的物件
        # print(f"*xyxy: {xyxy}, conf: {conf:.2f}, cls_id: {cls_id}")
        if(cls_id == 0 and conf >= 0.30):  # 如果檢測到的是人（cls_id為0）
            x1, y1, x2, y2 = map(int, xyxy)  # 獲取物件的座標
            centroid = ((x1 + x2) // 2, y2)  # 計算物件的中心點
            cv2.putText(frame, f'{(conf*100):.2f}%', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 在人物周圍畫矩形框
            
            found = False
            
            for trajectory in trajectories:  # 迭代現有軌跡
                if(np.linalg.norm(np.array(trajectory[-1]) - np.array(centroid)) < 10):  # 如果新中心點接近於已知軌跡
                    trajectory.append(centroid)  # 將新中心點添加到軌跡
                    new_trajectories.append(trajectory)
                    found = True
                    break

            if(not found):  # 如果沒有接近的軌跡
                new_trajectories.append([centroid])  # 創建新軌跡

            # print(trajectories)

    trajectories = new_trajectories  # 更新軌跡列表

    to_remove = []  # 初始化一個列表来存储需要刪除得軌跡得索引
    for i in range(len(trajectories)):  # 繪製所有軌跡
        cv2.arrowedLine(frame, trajectories[i][0], trajectories[i][-1], (0, 0, 255), 2, tipLength=0.1)
        cv2.polylines(frame, [np.array(trajectories[i], dtype=np.int32)], isClosed=False, color=(100, 100, 255), thickness=1)
        check_crossing = determine_crossing(trajectories[i][0], trajectories[i][-1], p1, p2)

        if check_crossing == 1:
            person_in_count += 1
            to_remove.append(i)
        elif check_crossing == -1:
            person_out_count += 1
            to_remove.append(i)

    # 刪除標記軌跡
    for index in sorted(to_remove, reverse=True):  # 逆續刪除以避免索引問題
        del trajectories[index]


    frame_count += 1  # fps計數器加一
    end_time = time.time()  # 記錄當前時間
    elapsed_time = end_time - start_time  # 計算經過時間
    actual_fps = frame_count / elapsed_time  # 計算實際幀率

    # 將計數和幀率信息顯示在畫面上
    cv2.putText(frame, f'in: {person_in_count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'out: {person_out_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'in - out: {person_in_count - person_out_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'fps: {actual_fps:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Video', frame)  # 顯示處理後的視頻幀
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):  # 如果按下 'q' 鍵，則退出
        break
    

cap.release()  # 釋放視頻捕獲對象
cv2.destroyAllWindows()  # 關閉所有 OpenCV 窗口