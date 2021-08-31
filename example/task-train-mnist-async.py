# Training the DL model with MNIST in the asynchronous mode
# @author: Trung Phan
# @created date: 2021-07-16
# @last modified date:
# @note:
import numpy as np
import time
from pympler import asizeof
from tensorflow.keras import datasets, layers, models, optimizers, utils
from ddlf.iapp import *
from ddlf.cluster import *


class App(IApp):
    def __init__(self):
        super().__init__()

    async def load_dataset(self):
        await self.cluster.run_method(self.__load_mnist)
        await self.__load_mnist()

    async def create_model(self):
        await self.cluster.run_method(self.__create_model)
        await self.__create_model()

    async def __load_mnist(self):
        from tensorflow.keras import datasets, utils
        # load the mnist dataset
        nb_classes = 10        
        (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.mnist.load_data()
        # Make sure images have shape (28, 28, 1)
        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)
        # Normalize pixel values from [0, 255] to [-0.5, 0.5] to to make it easier to work with
        self.x_train, self.x_test = (self.x_train / 255.0) - 0.5, (self.x_test / 255.0) - 0.5
        # Convert class vectors to binary class matrices (for using with loss='categorical_crossentropy')
        self.y_train = utils.to_categorical(self.y_train, nb_classes)
        self.y_test = utils.to_categorical(self.y_test, nb_classes)

    async def __create_model(self):
        nb_classes = 10
        input_shape = (28, 28, 1)
        # create the model
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(nb_classes, activation='softmax'))
        self.model.summary()
        self.model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model size: {asizeof.asizeof(self.model):,} bytes")

    # async def train_async1_old(self, master_epochs, epochs, batch_size):
    #     print("Asynchronous training...")
    #     await self.cluster.add_method(self.__train)
    #     await self.cluster.add_method(self.__load_partition)
    #
    #     lock = asyncio.Lock()
    #     # Shuffle the training data
    #     permutation = np.random.permutation(len(self.x_train))
    #     await self.cluster.run(self.__load_partition, permutation=permutation)
    #
    #     async def _train(worker):
    #         nonlocal lock, master_epochs, epochs, batch_size
    #         async with lock:
    #             _weights = self.model.get_weights()
    #         epoch = 0
    #         while (epoch < master_epochs):
    #             # train the model
    #             gradients = await self.cluster.run_at(worker, self.__train, weights=_weights, epochs=epochs,
    #                                                   batch_size=batch_size)
    #             # update weights of the model
    #             async with lock:
    #                 _weights = self.model.get_weights()
    #                 _weights = subtract(_weights, gradients)
    #                 self.model.set_weights(_weights)
    #             epoch += epochs
    #
    #     await asyncio.wait([_train(worker) for worker in self.cluster.workers])

    # async def train_async2_old(self, master_epochs, epochs, batch_size):
    #     print("Asynchronous training...")
    #     await self.cluster.add_method(self.__train)
    #     await self.cluster.add_method(self.__load_partition)
    #
    #     lock = asyncio.Lock()
    #     # Shuffle the training data
    #     permutation = np.random.permutation(len(self.x_train))
    #     await self.cluster.run(self.__load_partition, permutation=permutation)
    #
    #     async def _train(worker):
    #         nonlocal lock, master_epochs, epochs, batch_size
    #         async with lock:
    #             _weights = self.model.get_weights()
    #         epoch = 0
    #         # train the model
    #         gradients = await self.cluster.run_on(worker, self.__train, weights=_weights, epochs=master_epochs,
    #                                               batch_size=batch_size)
    #         # update weights of the model
    #         async with lock:
    #             _weights = self.model.get_weights()
    #             _weights = subtract(_weights, gradients)
    #             self.model.set_weights(_weights)
    #
    #     await asyncio.wait([_train(worker) for worker in self.cluster.workers])


async def main():
    app = App()
    await app.connect()
    await app.load_dataset()
    await app.create_model()
    await app.train_async(master_epochs=4, worker_epochs=2, batch_size=64)
    await app.close()

asyncio.run(main())
