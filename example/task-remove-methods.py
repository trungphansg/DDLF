# Training the DL model with MNIST in the asynchronous mode
# @author: Trung Phan
# @created date: 2021-07-18
# @last modified date:
# @note:
from ddltrain.cluster import *

async def f(self, lst):
    return sum(lst)

async def main():
    cluster = Cluster()
    await cluster.connect()
    await cluster.add_method(f)
    res = await cluster.run(f, lst=[1,2,3])
    print(res)
    await cluster.remove_method(f)
    res = await cluster.run(f, lst=[1, 2, 3]) # error
    await cluster.close()

asyncio.run(main())
