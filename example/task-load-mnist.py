# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddltrain.cluster import *


async def main():
    cluster = Cluster()
    await cluster.connect()
    await cluster.load_mnist()
    await cluster.close()


asyncio.run(main())
