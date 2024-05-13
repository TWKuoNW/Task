import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

cap = cv2.VideoCapture('D:/video/t1.MOV')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)  # 調整畫面的大小
    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', click_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()