# Training the DL model with MNIST in the asynchronous mode
# @author: Trung Phan
# @created date: 2021-07-18
# @last modified date:
# @note:
import numpy as np
import time
from pympler import asizeof
# from tensorflow.keras import datasets, layers, models, optimizers, utils
from ddltrain.iapp import *
from ddltrain.cluster import *


class App(IApp):
    def __init__(self):
        super().__init__()

    async def load_dataset(self):
        await self.cluster.run_method(self.__load_mnist)
        await self.__load_mnist()

    async def create_model(self):
        await self.cluster.run_method(self.__create_model)
        await self.__create_model()

    async def train1(self, master_epochs, epochs, batch_size):
        # perform distributed training
        permutation = np.random.permutation(len(self.x_train))
        await self.cluster.run_method(self.__load_partition, permutation=permutation)
        start_time = time.perf_counter()
        weights = self.model.get_weights()
        gradients = await self.cluster.run_method(self.__train, weights=weights, epochs=master_epochs, batch_size=batch_size)
        end_time = time.perf_counter() - start_time

        # update weights of the model
        for grad in gradients:
            weighted_grad = divide(grad, self.cluster.N)
            weights = subtract(weights, weighted_grad)
        self.model.set_weights(weights)

        score = self.model.evaluate(self.x_test, self.y_test)
        print("Epochs,Training Time,Loss,Accuracy")
        print(f"{master_epochs},{end_time},{score[0]},{score[1]}")

    async def train2(self, master_epochs, epochs, batch_size):
        # perform distributed training
        epoch = 0
        start_time = time.perf_counter()
        weights = self.model.get_weights()
        while (epoch < master_epochs):
            # Shuffle the training data
            permutation = np.random.permutation(len(self.x_train))
            await self.cluster.run_method(self.__load_partition, permutation=permutation)
            # train the model
            gradients = await self.cluster.run_method(self.__train, weights=weights, epochs=epochs,
                                                      batch_size=batch_size)
            # update weights of the model
            for grad in gradients:
                weighted_grad = divide(grad, self.cluster.N)
                weights = subtract(weights, weighted_grad)
            self.model.set_weights(weights)
            epoch += epochs
        end_time = time.perf_counter() - start_time

        score = self.model.evaluate(self.x_test, self.y_test)
        print("Epochs,Training Time,Loss,Accuracy")
        print(f"{master_epochs},{end_time},{score[0]},{score[1]}")

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

    async def __load_partition(self, permutation):
        try:
            n = partition_size = len(self.x_train) // self.N
            i = self.id
            self.x_train_partition = self.x_train[permutation][(i - 1) * n:i * n]
            self.y_train_partition = self.y_train[permutation][(i - 1) * n:i * n]
            return self.host, 'OK'
        except Exception as e:
            return self.host, e

    async def __train(self, weights, epochs, batch_size):
        self.model.set_weights(weights)
        self.model.fit(self.x_train_partition, self.y_train_partition, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
        new_weights = self.model.get_weights()
        gradients = subtract(weights, new_weights)
        return gradients


async def main():
    app = App()
    await app.connect()
    await app.load_dataset()
    await app.create_model()
    await app.train1(master_epochs=2, epochs=2, batch_size=64)
    await app.close()

asyncio.run(main())
