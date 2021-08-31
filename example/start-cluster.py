# Distributed DL Server runs on worker nodes
# @author: Trung Phan
# @created date: 2021-07-19
# @last modified date:
# @note:
from multiprocessing import Process
from ddltrain.worker import *


def main(i):
    worker = Worker(id=i+1, host='localhost', N=3, port=8888+i)
    worker.start()

processes = []
for i in range(3):
    p = Process(target=main, args=[i])
    processes.append(p)

if __name__ == '__main__':
    for p in processes:
        p.start()
    for p in processes:
        p.join()