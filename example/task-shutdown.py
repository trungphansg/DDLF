# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date: 2021-08-22
# @note:
from ddltrain.cluster import *


async def main():
    cluster = Cluster()
    await cluster.connect()
    await cluster.shutdown()


asyncio.run(main())
