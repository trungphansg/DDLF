# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddltrain.cluster import *
from ddltrain.app import *


async def main(N=9):
    app = App()
    await app.connect()
    await app.load_dataset()
    await app.train1(epochs=10, batch_size=64)
    await app.train2(master_epochs=10, epochs=2, batch_size=64)
    await app.close()


asyncio.run(main())
