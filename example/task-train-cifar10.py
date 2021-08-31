# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddlf.cluster import *



async def main(N=9):
    app = App()
    await app.connect()
    await app.load_cifar10()
    await app.train_cifar10(master_epochs=100, epochs=10, batch_size=64)
    await app.close()


asyncio.run(main())
