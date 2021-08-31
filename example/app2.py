# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
import numpy as np
import time
from pympler import asizeof
from tensorflow.keras import datasets, layers, models, optimizers, utils

from ddlf.cluster import *
from ddlf.tools import *

class App2(object):
    def __init__(self, cluster: Cluster, N: int):
        self.cluster = cluster
        self.N = N
        
    async def connect(self):
        await self.cluster.connect()

    async def close(self):
        await self.cluster.close()
        
    async def load_dataset(self):
        await self.cluster.load_dataset()
        
    async def load_mnist(self):
        await self.cluster.load_mnist()

    async def load_cifar10(self):
        await self.cluster.load_cifar10()
        
    async def master_load_dataset(self):
        dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
        self.x_train = dataset[:, 0:8]
        self.y_train = dataset[:, 8]
        self.x_test = dataset[0:10, 0:8]
        self.y_test = dataset[0:10, 8]


    async def train1(self, epochs, batch_size):
        # load dataset
        (x_train, y_train), (x_test, y_test) = await self.master_load_dataset()
        # create a model
        model = models.Sequential()
        model.add(layers.Dense(12, input_dim=8, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        weights = model.get_weights()

        start_time = time.perf_counter_ns()
        # Schedule three calls *concurrently*:
        await self.cluster.create_model(yaml=model.to_yaml(), loss='binary_crossentropy', optimizer='adam',
                                  metrics=['accuracy'])
        # Shuffle the training data
        permutation = np.random.permutation(len(x_train))
        # perform distributed training
        gradients = await self.cluster.train(weights, epochs, batch_size, permutation)
        # update weights of the model
        for grad in gradients:
            weighted_grad = divide(grad, self.N)
            weights = subtract(weights, weighted_grad)
        model.set_weights(weights)
        print(f"Time: {time.perf_counter_ns() - start_time:,} nanoseconds")
        print(f"Time: {(time.perf_counter_ns() - start_time) // (10 ** 6):,} milliseconds")
        # evaluate the model
        score = model.evaluate(x_test, y_test)
        print('Final test scores [loss,acc]:', score)
        print(f"Model size: {asizeof.asizeof(model):,}")

    async def train2(self, master_epochs, epochs, batch_size):
        # load dataset
        (x_train, y_train), (x_test, y_test) = await self.master_load_dataset()
        # create a model
        model = models.Sequential()
        model.add(layers.Dense(12, input_dim=8, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        weights = model.get_weights()

        start_time = time.perf_counter_ns()
        # Schedule three calls *concurrently*:
        await self.cluster.create_model(yaml=model.to_yaml(), loss='binary_crossentropy', optimizer='adam',
                                  metrics=['accuracy'])
        # perform distributed training
        count = 0
        while (count < master_epochs):
            # Shuffle the training data
            permutation = np.random.permutation(len(x_train))
            # train the model
            gradients = await self.cluster.train(weights, epochs, batch_size, permutation)
            # update weights of the model
            for grad in gradients:
                weighted_grad = divide(grad, self.N)
                weights = subtract(weights, weighted_grad)
            model.set_weights(weights)
            count += epochs
        print(f"Time: {time.perf_counter_ns() - start_time:,} nanoseconds")
        print(f"Time: {(time.perf_counter_ns() - start_time) // (10 ** 6):,} milliseconds")
        # evaluate the model
        score = model.evaluate(x_test, y_test)
        print('Final test scores [loss,acc]:', score)
        print(f"Model size: {asizeof.asizeof(model):,}")

    async def train_mnist(self, master_epochs, epochs, batch_size):
        nb_classes = 10
        input_shape = (28, 28, 1)
        # load the mnist dataset
        try:
            (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
            # Make sure images have shape (28, 28, 1)
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            # Normalize pixel values from [0, 255] to [-0.5, 0.5] to to make it easier to work with
            x_train, x_test = (x_train / 255.0) - 0.5, (x_test / 255.0) - 0.5
            # Convert class vectors to binary class matrices (for using with loss='categorical_crossentropy')
            y_train = utils.to_categorical(y_train, nb_classes)
            y_test = utils.to_categorical(y_test, nb_classes)
        except Exception as e:
            print(e)
        # create the model
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(nb_classes, activation='softmax'))
        model.summary()
        # model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        weights = model.get_weights()
        print(f"Weights size: {asizeof.asizeof(weights):,} bytes")

        start_time = time.perf_counter_ns()
        # Schedule three calls *concurrently*:
        # await self.cluster.create_model(yaml=model.to_yaml(), loss='categorical_crossentropy', optimizer=SGD(lr=0.1),
        #                           metrics=['accuracy'])
        await self.cluster.create_model(yaml=model.to_yaml(), loss='categorical_crossentropy', optimizer='SGD',
                                        metrics=['accuracy'])
        # perform distributed training
        count = 0
        while (count < master_epochs):
            # Shuffle the training data
            permutation = np.random.permutation(len(x_train))
            # train the model
            gradients = await self.cluster.train(weights, epochs, batch_size, permutation)
            # update weights of the model
            for grad in gradients:
                weighted_grad = divide(grad, self.N)
                weights = subtract(weights, weighted_grad)
            model.set_weights(weights)
            count += epochs
            score = model.evaluate(x_test, y_test)
            print(f'Epoch {count}: scores[loss,acc]:{score}')

        print(f"Time: {time.perf_counter_ns() - start_time:,} nanoseconds")
        print(f"Time: {(time.perf_counter_ns() - start_time) // (10 ** 6):,} milliseconds")
        # evaluate the model
        score = model.evaluate(x_test, y_test)
        print('Final test scores [loss,acc]:', score)
        print(f"Model size: {asizeof.asizeof(model):,} bytes")

    async def train_cifar10(self, master_epochs, epochs, batch_size):
        nb_classes = 10
        input_shape = (32, 32, 3)
        # load the cifar10 dataset
        try:
            # cifar image features
            (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
            # Normalize pixel values from [0, 255] to [-0.5, 0.5] to to make it easier to work with
            x_train, x_test = (x_train / 255.0) - 0.5, (x_test / 255.0) - 0.5
            # Convert class vectors to binary class matrices (for using with loss='categorical_crossentropy')
            y_train = utils.to_categorical(y_train, nb_classes)
            y_test = utils.to_categorical(y_test, nb_classes)
        except Exception as e:
            print(e)
        # create the model
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(nb_classes, activation='softmax'))
        model.summary()
        # model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        weights = model.get_weights()
        print(f"Weights size: {asizeof.asizeof(weights):,} bytes")

        start_time = time.perf_counter_ns()
        # Schedule three calls *concurrently*:
        # await self.cluster.create_model(yaml=model.to_yaml(), loss='categorical_crossentropy', optimizer=SGD(lr=0.1),
        #                           metrics=['accuracy'])
        await self.cluster.create_model(yaml=model.to_yaml(), loss='categorical_crossentropy', optimizer='SGD',
                                        metrics=['accuracy'])
        # perform distributed training
        count = 0
        while (count < master_epochs):
            # Shuffle the training data
            permutation = np.random.permutation(len(x_train))
            # train the model
            gradients = await self.cluster.train(weights, epochs, batch_size, permutation)
            # update weights of the model
            for grad in gradients:
                weighted_grad = divide(grad, self.N)
                weights = subtract(weights, weighted_grad)
            model.set_weights(weights)
            count += epochs
            score = model.evaluate(x_test, y_test)
            print(f'Epoch {count}: scores[loss,acc]:{score}')

        print(f"Time: {time.perf_counter_ns() - start_time:,} nanoseconds")
        print(f"Time: {(time.perf_counter_ns() - start_time) // (10 ** 6):,} milliseconds")
        # evaluate the model
        score = model.evaluate(x_test, y_test)
        print('Final test scores [loss,acc]:', score)
        print(f"Model size: {asizeof.asizeof(model):,} bytes")