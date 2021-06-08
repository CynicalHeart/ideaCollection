# 多线程

import os
import random
import threading
import time
from multiprocessing import Pool, Process


def loop():
    # 新线程执行的代码:
    print('thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)


def thread_test():
    # 线程测试
    print('thread %s is running...' % threading.current_thread().name)
    t = threading.Thread(target=loop, name='LoopThread')
    t.start()
    t.join()
    print('thread %s ended.' % threading.current_thread().name)


def long_time_task(name):
    # 如果要启动大量的子进程，可以用进程池的方式批量创建子进程
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))


def run_proc(name):
    # 子进程要执行的代码
    print('Run child process %s (%s)...' % (name, os.getpid()))


def test_process():
    print('Parent process %s.' % os.getpid())  # 父进程
    p = Process(target=run_proc, args=('test',))  # 执行函数和传入的参数
    print('Child process will start.')
    p.start()  # 启动进程
    p.join()  # 等待子进程结束后再继续往下运行，通常用于进程间的同步。
    print('Child process end.')


def test_pool():
    print('Parent process %s.' % os.getpid())
    p = Pool(4)  # 4进程池
    for i in range(5):  # 5个任务进程
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()  # 调用join()之前必须先调用close()，调用close()之后就不能继续添加新的Process了。
    p.join()  # 对Pool对象调用join()方法会等待所有子进程执行完毕
    print('All subprocesses done.')


if __name__ == '__main__':
    # test_process()  # 少量子进程
    # test_pool()  # 进程池
    # thread_test()
    import multiprocessing
    print(multiprocessing.cpu_count())
