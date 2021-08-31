# Training the DL model with MNIST in the asynchronous mode
# @author: Trung Phan
# @created date: 2021-07-18
# @last modified date:
# @note:
from ddlf.cluster import *


async def f(self, lst):
    return sum(lst)


async def main():
    cluster = Cluster()
    await cluster.connect()
    await cluster.add_method(f)
    res = await cluster.run_on(cluster.workers[0], f, lst=[1, 2, 3])
    print(res)
    res = await cluster.run_on(cluster.workers[1], f, lst=[3, 4, 5])
    print(res)
    await cluster.close()


asyncio.run(main())
