# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddltrain.cluster import *


async def calculate(self, a, b, c):
    return a + b - c


async def main():
    cluster = Cluster()
    await cluster.connect()
    results = await cluster.run_method(calculate, a=10, b=8, c=2)
    print(f"Results: {results}") # Results: [16, 16, 16]
    results = await cluster.run_method(calculate, dict(a=10, b=8, c=2), dict(a=100, b=80, c=20), dict(a=1000, b=800, c=200))
    print(f"Results: {results}") # Results: [16, 160, 1600]
    results = await cluster.run_method(calculate, dict(a=10, b=8, c=2), dict(a=100, b=80, c=20))
    print(f"Results: {results}") # Results: [16, 160]
    await cluster.close()


asyncio.run(main())
