# IApp is an interface for training DL models in the distributed mode
# @author: Trung Phan
# @created date: 2021-07-17
# @last modified date: 2021-08-22
# @note:
import numpy as np
import time
from ddlf.cluster import *


class IApp(IWorker):
    def __init__(self):
        self.cluster = Cluster()
        self.N = self.cluster.N
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    async def connect(self):
        await self.cluster.connect()

    async def clean(self):
        await self.cluster.clean()

    async def close(self):
        await self.cluster.close()

    async def create_model(self):
        raise NotImplemented

    async def load_dataset(self):
        raise NotImplemented

    async def shutdown(self):
        await self.cluster.shutdown()

    async def __train(self, weights, worker_epochs, batch_size):
        self.model.set_weights(weights)
        self.model.fit(self.x_train_partition, self.y_train_partition, epochs=worker_epochs, batch_size=batch_size,
                       validation_split=0.1, verbose=2)
        new_weights = self.model.get_weights()
        gradients = subtract(weights, new_weights)
        gradients = divide(gradients, self.N)
        return gradients

    # perform asynchronous distributed training
    async def train_async(self, master_epochs, worker_epochs, batch_size):
        data_size = len(self.x_train)
        print(f"Data size: {data_size}")
        await self.cluster.add_method(self.__train)
        await self.cluster.add_method(self.__load_partition)

        lock = asyncio.Lock()
        async def _train(worker):
            nonlocal lock, master_epochs, worker_epochs, batch_size, data_size
            async with lock:
                _weights = self.model.get_weights()
            epoch = 0
            while (epoch < master_epochs):
                # Shuffle the training data
                permutation = np.random.permutation(data_size)
                await self.cluster.run_on(worker, self.__load_partition, permutation=permutation)
                # train the model
                gradients = await self.cluster.run_on(worker, self.__train, weights=_weights, worker_epochs=worker_epochs,
                                                      batch_size=batch_size)
                # update weights of the model
                async with lock:
                    _weights = self.model.get_weights()
                    _weights = subtract(_weights, gradients)
                    self.model.set_weights(_weights)
                epoch += worker_epochs

        await asyncio.wait([_train(worker) for worker in self.cluster.workers])

    # perform synchronous distributed training
    async def train_sync(self, master_epochs, worker_epochs, batch_size):
        data_size = len(self.x_train)
        await self.cluster.add_method(self.__train)
        await self.cluster.add_method(self.__load_partition)
        epoch = 0
        start_time = time.perf_counter()
        weights = self.model.get_weights()
        while (epoch < master_epochs):
            # Shuffle the training data
            permutation = np.random.permutation(data_size)
            await self.cluster.run(self.__load_partition, permutation=permutation)
            # train the model
            gradients = await self.cluster.run(self.__train,
                                                      weights=weights,
                                                      worker_epochs=worker_epochs,
                                                      batch_size=batch_size)
            # update weights of the model
            for gradient in gradients:
                weights = subtract(weights, gradient)
            self.model.set_weights(weights)
            epoch += worker_epochs
        end_time = time.perf_counter() - start_time

        score = self.model.evaluate(self.x_test, self.y_test)
        print("Epochs,Training Time,Loss,Accuracy")
        print(f"{master_epochs},{end_time},{score[0]},{score[1]}")

    async def __load_partition(self, permutation):
        try:
            n = partition_size = len(self.x_train) // self.N
            i = self.id
            self.x_train_partition = self.x_train[permutation][(i - 1) * n:i * n]
            self.y_train_partition = self.y_train[permutation][(i - 1) * n:i * n]
            return None
        except Exception as e:
            return e