# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddlf.cluster import *


async def main():
    cluster = Cluster()
    await cluster.connect()
    for i in range(3):
        await cluster.ping()
    await cluster.close()


asyncio.run(main())
