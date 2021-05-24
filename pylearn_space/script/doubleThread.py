# 双线程opencv切换视频
import cv2
from pathlib import Path
from multiprocessing import Process


def local_video(path: str):
    cap = cv2.VideoCapture(path)
    while(True):
        ret, frame = cap.read()
        if ret is True:
            cv2.imshow(Path(path).name.split('.')[0], frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


if __name__ == '__main__':
    path_1 = "D:\\workspace\\MTMC\\video\\test_01.mp4"  # 路径1
    path_2 = "D:\\workspace\\MTMC\\video\\test_02.mp4"  # 路径2
    p1 = Process(target=local_video, args=(path_1,))
    p2 = Process(target=local_video, args=(path_2,))
    p1.start()
    p2.start()
    # 只能阻塞, 否则会和mian线程混合. 这种方法只适合做简单的demo
    p1.join()
    p2.join()

    cv2.destroyAllWindows()
