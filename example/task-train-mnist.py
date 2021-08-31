# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddltrain.cluster import *
from ddltrain.app import *


async def main(N=9):
    # hosts = [f'hadoop{id}' for id in range(2, N + 1)]
    # ports = [8888 for id in range(2, N + 1)]
    # hosts = ['192.168.146.1', '192.168.44.1', '192.168.1.8']
    # ports = [8888 for id in range(3)]
    hosts = ['localhost' for i in range(3)]
    ports = [8888+i for i in range(3)]
    app = App(Cluster(hosts, ports), len(hosts))
    await app.connect()
    await app.load_mnist()
    await app.train_mnist(master_epochs=1, epochs=1, batch_size=64)
    await app.close()

asyncio.run(main())
