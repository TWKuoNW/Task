import cv2  
import torch 
import numpy as np  
import time  

def determine_crossing(A, B, C, D): # 判斷是否有人進入或離開區域
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

def find_matching_path(current_paths, reference_path): # 找到匹配的路徑
    for path in current_paths:
        if path[:len(reference_path)] == reference_path:
            return path
    return None

# 加載 yolov5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # 設置模型模式
cap = cv2.VideoCapture('D:/video/t4.MOV')

trajectories = []  # 初始化路徑列表
person_in_count = 0  # 初始化進入區域的人數
person_out_count = 0  # 初始化離開區域的人數
frame_count = 0  # 初始化fps計數器
"""
# 定義一個區域的四個點
p1 = [0, 625] # 左上
p2 = [1366, 600] # 右上
p3 = [1366, 768] # 右下
p4 = [0, 768] # 左下
"""
p1 = [0, 635] # 左上
p2 = [1366, 650] # 右上
p3 = [1366, 768] # 右下
p4 = [0, 768] # 左下
points = np.array([p1, p2, p3, p4], dtype=np.int32) 

# 啟用錄影功能
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('D:/video/outputV2_4.avi', fourcc, 20.0, (1366, 768))

start_time = time.time()  # 記錄開始處理時間

while(cap.isOpened()):
    ret, frame = cap.read()  # 讀取一幀
    if(not ret):
        break  # 如果無法讀取，跳出循環

    frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)  # 調整畫面的大小
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 將幀轉換成 RGB
    results = model([img], size=640)  # 使用模型進行預測
    results = results.xyxy[0].numpy()  # 提取結果並轉換為 NumPy Array

    cv2.polylines(frame, [points], isClosed=True, color=(255, 100, 100), thickness=2)  # 繪製多邊形

    overlay = np.zeros_like(frame) # 創建一個與幀大小相同的透明層
    cv2.fillPoly(overlay, [points], color=(255, 100, 100)) # 填充多邊形區域
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0) # 添加透明層

    new_trajectories = []  # 初始化新的軌跡列表
    disappeared_paths = [] # 儲存在新一幀中未找到的路徑

    for *xyxy, conf, cls_id in results:  # 迭代檢測到的物件
        # print(f"*xyxy: {xyxy}, conf: {conf:.2f}, cls_id: {cls_id}")
        if(cls_id == 0 and conf >= 0.30):  # 如果檢測到的是人（cls_id為0）
            x1, y1, x2, y2 = map(int, xyxy)  # 獲取物件的座標
            centroid = ((x1 + x2) // 2, y2)  # 計算物件的中心點
            cv2.putText(frame, f'{(conf*100):.2f}%', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # 在物件上顯示可信度
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

    # 檢查每個舊路徑在新幀中是否有匹配
    for old_path in trajectories:
        if not find_matching_path(new_trajectories, old_path):
            disappeared_paths.append(old_path)

    # 輸出消失的路徑，並且辨識進入或離開
    for path in disappeared_paths:
        check_crossing = determine_crossing(path[0], path[-1], p1, p2)
        if check_crossing == 1:
            person_in_count += 1
        elif check_crossing == -1:
            person_out_count += 1

    trajectories = new_trajectories  # 更新路徑列表

    for trajectory in trajectories:  # 迭代所有軌跡
        cv2.arrowedLine(frame, trajectory[0], trajectory[-1], (0, 0, 255), 2, tipLength=0.1)  # 繪製箭頭
        cv2.polylines(frame, [np.array(trajectory, dtype=np.int32)], isClosed=False, color=(100, 100, 255), thickness=1)  # 繪製軌跡

    frame_count += 1  # fps計數器加一
    end_time = time.time()  # 記錄當前時間
    elapsed_time = end_time - start_time  # 計算經過時間
    actual_fps = frame_count / elapsed_time  # 計算實際幀率

    # 將計數和幀率信息顯示在畫面上
    cv2.putText(frame, f'in: {person_in_count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'out: {person_out_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'in - out: {person_in_count - person_out_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'fps: {actual_fps:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Video', frame)  # 顯示處理後的影像幀
    out.write(frame) # 寫入影像
    print(frame)

    if cv2.waitKey(1) == ord('q'):  # 如果按下 'q' 鍵，則退出
        break
    
cap.release()  # 釋放 cap
cv2.destroyAllWindows()  # 關閉所有 OpenCV 窗口
print("Done!")