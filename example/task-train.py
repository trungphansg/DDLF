# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddlf.cluster import *
from app import *


async def main():
    app = App()
    await app.connect()
    await app.load_dataset()
    await app.train1(epochs=10, batch_size=64)
    await app.train2(master_epochs=10, epochs=2, batch_size=64)
    await app.close()


asyncio.run(main())
