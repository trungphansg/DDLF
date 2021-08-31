# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddltrain.cluster import *
import time


async def send(self, weights, start_time):
    self.data['weights'] = weights
    end_time = time.time_ns()
    return end_time - start_time


async def main():
    cluster = Cluster()
    await cluster.connect()
    weights=b'1'*2061616
    res = await cluster.run_method(send, weights=weights, start_time=time.time_ns())
    print(f"result: {res}")
    await cluster.close()


asyncio.run(main())
