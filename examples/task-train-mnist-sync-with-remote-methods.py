# Training the DL model with MNIST in the synchronous mode
# using remote methods
# @author: Trung Phan
# @created date: 2021-08-31
# @last modified date:
# @note:
import numpy as np
import time
from pympler import asizeof
# from tensorflow.keras import datasets, layers, models, optimizers, utils
from ddlf.iapp import *
from ddlf.cluster import *


class App(IApp):
    def __init__(self):
        super().__init__()

    async def load_dataset(self):
        await self.cluster.load_mnist()

    async def create_model(self):
        await self.cluster.run_method(self.__create_model)
        await self.__create_model()

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

    # perform synchronous distributed training
    async def train_sync(self, master_epochs, worker_epochs, batch_size):
        data_size = 60000
        epoch = 0
        start_time = time.perf_counter()
        weights = self.model.get_weights()
        while (epoch < master_epochs):
            # Shuffle the training data
            permutation = np.random.permutation(data_size)
            await self.cluster.load_partition(permutation=permutation)
            # train the model
            gradients = await self.cluster.train(weights, worker_epochs, batch_size)
            # update weights of the model
            for gradient in gradients:
                weights = subtract(weights, gradient)
            self.model.set_weights(weights)
            epoch += worker_epochs
        end_time = time.perf_counter() - start_time

        # score = self.model.evaluate(self.x_test, self.y_test)
        # print("Epochs,Training Time,Loss,Accuracy")
        # print(f"{master_epochs},{end_time},{score[0]},{score[1]}")

async def main():
    app = App()
    await app.connect()
    await app.load_dataset()
    await app.create_model()
    await app.train_sync(master_epochs=4, worker_epochs=2, batch_size=64)
    await app.close()

asyncio.run(main())
