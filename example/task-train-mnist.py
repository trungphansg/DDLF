# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddlf.cluster import *



async def main():
    app = App()
    await app.connect()
    await app.load_mnist()
    await app.train_mnist(master_epochs=1, epochs=1, batch_size=64)
    await app.close()

asyncio.run(main())
