# 双视频交错跑
import cv2 as cv

path_1 = "D:\\workspace\\MTMC\\video\\test_01.mp4"  # 路径1
path_2 = "D:\\workspace\\MTMC\\video\\test_02.mp4"  # 路径2

cap_1 = cv.VideoCapture(path_1)  # 摄像机1
cap_2 = cv.VideoCapture(path_2)  # 摄像机2

# cv.namedWindow('cam-1', cv.WINDOW_NORMAL)  # 窗口1(可调节)
# cv.namedWindow('cam-2', cv.WINDOW_NORMAL)  # 窗口2

while True:

    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()

    if (not ret_1) and (not ret_2):
        # 都结束退出
        print('ending!')
        break

    elif not ret_2:
        cv.imshow('cam-1', frame_1)

    elif not ret_1:
        cv.imshow('cam-2', frame_2)
    else:
        cv.imshow('cam-1', frame_1)
        cv.imshow('cam-2', frame_2)

    if cv.waitKey(10) & 0xff == ord('q'):
        print('exit')
        break

cap_1.release()
cap_2.release()
cv.destroyAllWindows()
