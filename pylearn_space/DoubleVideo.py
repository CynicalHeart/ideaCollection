# 双视频同时跑
import cv2 as cv

path_1 = "G:/resources/拍摄视频/sau1.mp4"
path_2 = "G:/resources/拍摄视频/test_01.mp4"
cap_1 = cv.VideoCapture(path_1)  # 摄像机1
cap_2 = cv.VideoCapture(path_2)  # 摄像机2

cv.namedWindow('cam-1')  # 窗口1
cv.namedWindow('cam-2')  # 窗口2

while True:
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()

    if (not ret_1) and (not ret_2):
        print('ending...')
        break

    cv.imshow('cam-1', frame_1)
    cv.imshow('cam-2', frame_2)

    if cv.waitKey(30) & 0xff == ord('q'):
        print('exit')
        break

cap_1.release()
cap_2.release()
cv.destroyAllWindows()
