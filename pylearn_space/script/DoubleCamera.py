# 双摄相机调用测试
import cv2 as cv


cap_1 = cv.VideoCapture(0)  # 摄像机1
cap_2 = cv.VideoCapture(1)  # 摄像机2

cv.namedWindow('cam-1')  # 窗口1
cv.namedWindow('cam-2')  # 窗口2

while True:
    ret, frame_1 = cap_1.read()
    ret, frame_2 = cap_2.read()

    cv.imshow('cam-1', frame_1)
    cv.waitKey(10)
    cv.imshow('cam-2', frame_2)

    if cv.waitKey(10) & 0xff == ord('q'):
        print('exit')
        break

cap_1.release()
cap_2.release()
cv.destroyAllWindows()
