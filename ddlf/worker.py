# Distributed DL Server runs on worker nodes
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date: 2021-09-03
# @note:
import asyncio
import gc
import numpy as np
import sys
import time
from asyncio import StreamReader, StreamWriter
from pympler import asizeof
from tensorflow.keras import datasets, layers, models, optimizers, utils
from tensorflow.keras.models import *
from textwrap import dedent
from ddlf.iworker import *
from ddlf.request import *
from ddlf.response import *
from ddlf.status import *
from ddlf.tools import *
from ddlf.transport import *


class Worker(IWorker):
    def __init__(self, id: int=1, N: int=1, host='localhost', port=8888):
        self.id = id
        self.N = N #number of workers
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.n = 80 # the length of separators
        # data can clean
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_train_partition = None
        self.y_train_partition = None
        self.model: Sequential = None
        self.data = {} #used to store any data

    def start(self):
        self.loop = asyncio.get_event_loop()
        self.server = self.loop.run_until_complete(asyncio.start_server(self.control, self.host, self.port))
        print(f'The worker {self.id} ({self.host}:{self.port}) is already...')
        try:
            self.loop.run_forever()
        except KeyboardInterrupt as e:
            print(e)
        except Exception as e:
            print(e)
        self.loop.close()
        print(f'The worker {self.id} ({self.host}:{self.port}) has shut down. See you again!')

    async def control(self, reader: StreamReader, writer: StreamWriter):
        peer = writer.get_extra_info('peername')
        print('-' * self.n, f"\nOpen the connection with {peer}")
        shutdown_flag = False
        loop = True
        while loop:
            # receive a request
            req = await recv_message(reader)
            # handle the request
            loop, res, shutdown_flag = await self.handle(req)
            # show the result
            # print(f'cmd:{req.command}, response:{res}')
            # send the result to the client
            await send_message(res, writer)
        # close the connection
        print('-' * self.n, f"\nClose the connection with {peer}")
        writer.close()

        if shutdown_flag:
            await self.shutdown()

    async def handle(self, req: Request):
        shutdown_flag = False
        loop = True
        # print(f"Handling the request {req.command!r}...")
        # processing the request
        if req.command == 'add_method':
            res = await self.add_method(**req.kwargs)
        if req.command == 'clean':
            res = await self.clean()
        elif req.command == 'close':
            loop = False
            res = await self.close()
        elif req.command == 'load_cifar10':
            res = await self.load_cifar10()
        elif req.command == 'load_mnist':
            res = await self.load_mnist()
        elif req.command == 'load_partition':
            res = await self.load_partition(**req.kwargs)
        elif req.command == 'ping':
            res = await self.ping()
        elif req.command == 'remove_method':
            res = await self.remove_method(method_name=req.args[0])
        elif req.command == 'run':
            res = await self.run(method_name=req.args[0], **req.kwargs)
        elif req.command == 'run_code':
            res = await self.run_code(code=req.args[0])
        elif req.command == 'run_method':
            res = await self.run_method(method_code=req.args[0], method_name=req.args[1], **req.kwargs)
        elif req.command == 'show_data':
            res = await self.show_data()
        elif req.command == 'shutdown':
            shutdown_flag = True
            loop = False
            res = Response(Status.OK, None)
        elif req.command == 'train':
            res = await self.train(**req.kwargs)
        # print(f"Finished handling the request {req.command!r}.")
        return loop, res, shutdown_flag

    async def add_method(self, method_code, method_name):
        print('-'*self.n,f'\nExecuting add_method({method_name})...')
        # re-indentation
        method_code = dedent(method_code)
        # print(f"Method name: {method_name}")
        # print(f"Method length: {len(method_code)}")
        # print(f"Method code:\n{method_code}")
        try:
            code = f'''{method_code}\nsetattr(Worker, {method_name!r}, {method_name})'''
            exec(code)
            print(f'Finished executing add_method({method_name}).')
            return Response(Status.OK, None)
        except Exception as e:
            print(f'Exception when executing add_method({method_name}):', e)
            return Response(Status.ERROR, e)

    async def clean(self):
        print('-' * self.n, '\nExecuting clean()...')
        try:
            self.x_train = None
            self.y_train = None
            self.x_test = None
            self.y_test = None
            self.x_train_partition = None
            self.y_train_partition = None
            self.model: Sequential = None
            self.data.clear()
            gc.collect()
            print('Finished executing clean().')
            return Response(Status.OK, None)
        except Exception as e:
            print('Exception when executing clean():', e)
            return Response(Status.ERROR, e)

    async def close(self):
        return Response(Status.OK, None)

    async def load_cifar10(self):
        print('-' * self.n, '\nExecuting load_cifar10()...')
        nb_classes = 10
        input_shape = (32, 32, 3)
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.cifar10.load_data()
            # Normalize pixel values from [0, 255] to [-0.5, 0.5] to to make it easier to work with
            self.x_train, self.x_test = (self.x_train / 255.0) - 0.5, (self.x_test / 255.0) - 0.5
            # Convert class vectors to binary class matrices (for using with loss='categorical_crossentropy')
            self.y_train = utils.to_categorical(self.y_train, nb_classes)
            self.y_test = utils.to_categorical(self.y_test, nb_classes)
            print('Finished executing load_cifar10().')
            return Response(Status.OK, None)
        except Exception as e:
            print('Exception when executing load_cifar10():', e)
            return Response(Status.ERROR, e)

    async def load_mnist(self):
        print('-' * self.n, '\nExecuting load_mnist()...')
        nb_classes = 10
        input_shape = (28, 28, 1)
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = datasets.mnist.load_data()
            # Make sure images have shape (28, 28, 1)
            self.x_train = np.expand_dims(self.x_train, -1)
            self.x_test = np.expand_dims(self.x_test, -1)
            # Normalize pixel values from [0, 255] to [-0.5, 0.5] to to make it easier to work with
            self.x_train, self.x_test = (self.x_train / 255.0) - 0.5, (self.x_test / 255.0) - 0.5
            # Convert class vectors to binary class matrices (for using with loss='categorical_crossentropy')
            self.y_train = utils.to_categorical(self.y_train, nb_classes)
            self.y_test = utils.to_categorical(self.y_test, nb_classes)
            print('Finished executing load_mnist().')
            return Response(Status.OK, None)
        except Exception as e:
            print('Exception when executing load_mnist():', e)
            return Response(Status.ERROR, e)

    async def load_partition(self, permutation):
        """
        Load a partition of data for training the the model on a worker node.
        """
        try:
            n = partition_size = len(self.x_train) // self.N
            i = self.id
            self.x_train_partition = self.x_train[permutation][(i - 1) * n:i * n]
            self.y_train_partition = self.y_train[permutation][(i - 1) * n:i * n]
            return Response(Status.OK, None)
        except Exception as e:
            return Response(Status.ERROR, e)

    async def ping(self):
        print('-' * self.n, '\nExecuting ping()...')
        return Response(Status.OK, None)

    async def remove_method(self, method_name):
        print('-' * self.n, f'\nExecuting remove_method({method_name})...')
        try:
            delattr(Worker, method_name)
            print(f'Finished executing remove_method({method_name}).')
            return Response(Status.OK, None)
        except Exception as e:
            print(f'Exception when executing remove_method({method_name}):', e)
            return Response(Status.ERROR, e)

    async def run(self, method_name, **kwargs):
        # method_name = kwargs['method_name']
        # args = { key:value for (key,value) in kwargs.items() if key not in ['method_name']}
        print('-' * self.n, f'\nExecuting run({method_name})...')
        print(f"Method name: {method_name}")
        try:
            code = f'''setattr(Worker, '_method', self.{method_name})'''
            exec(code)
            result = await self._method(**kwargs)
            print(f'Finished executing run({method_name}).')
            return Response(Status.OK, result)
        except Exception as e:
            print(f'Exception when executing run({method_name}):', e)
            return Response(Status.ERROR, e)

    async def run_code(self, code):
        print('-' * self.n, '\nExecuting run_code()...')
        try:
            # re-indentation
            code = dedent(code)
            # print(f"Code length: {len(code)}")
            # print(f"Code:\n{code}")
            exec(code)
            print('Finished executing run_code().')
            return Response(Status.OK, None)
        except Exception as e:
            print('Exception when executing run_code():', e)
            return Response(Status.ERROR, e)

    async def run_method(self, method_code, method_name, **kwargs):
        print('-' * self.n, f'\nExecuting run_method({method_name})...')
        # re-indentation
        method_code = dedent(method_code)
        # print(f"Method name: {method_name}")
        # print(f"Method length: {len(method_code)}")
        # print(f"Method code:\n{method_code}")
        try:
            code = f'''{method_code}\nsetattr(Worker, '_method', {method_name})'''
            exec(code)
            result = await self._method(**kwargs)
            # locals = {}
            # code = f'''res = self.{method_name}(**kwargs)'''
            # exec(code, None, locals)
            print(f'Finished executing run_method({method_name}).')
            return Response(Status.OK, result)
        except Exception as e:
            print(f'Exception when executing run_method({method_name}):', e)
            return Response(Status.ERROR, e)

    async def show_data(self):
        print('-' * self.n, '\nExecuting show_data()...')
        return Response(Status.OK, self.data)

    async def shutdown(self):
        print('-' * self.n, '\nExecuting shutdown()...')
        await asyncio.sleep(0)
        print("The master requests shutdown")
        self.server.close()
        print("The internal server has closed")
        self.loop.stop()
        print("The event loop has stopped")
        print('Finished executing shutdown().')

    async def train(self, weights, worker_epochs, batch_size):
        """
        Train the model on a worker node.
        """
        try:
            self.model.set_weights(weights)
            self.model.fit(self.x_train_partition, self.y_train_partition,
                           epochs=worker_epochs, batch_size=batch_size,
                           validation_split=0.1, verbose=2)
            new_weights = self.model.get_weights()
            gradients = subtract(weights, new_weights)
            gradients = divide(gradients, self.N)
            return Response(Status.OK, gradients)
        except Exception as e:
            print(f'Exception when executing train(weights={weights}, worker_epochs={worker_epochs}, batch_size={batch_size}):', e)
            return Response(Status.ERROR, e)

# worker = Worker(id=int(sys.argv[1]), host=sys.argv[2], N=int(sys.argv[3]))
# worker.start()

# for test only
# worker = Worker(id=1, host='localhost', N=1)
# worker.start()

# from multiprocessing import Process
#
# def main(i):
#     # hosts = ['192.168.146.1', '192.168.44.1', '192.168.1.8']
#     worker = Worker(id=i+1, host='localhost', N=3, port=8888+i)
#     worker.start()
#     # worker = Worker(id=i+1, host=hosts[i], N=3)
#     # worker.start()
#
# processes = []
# for i in range(3):
#     p = Process(target=main, args=[i])
#     processes.append(p)
#
# if __name__ == '__main__':
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()