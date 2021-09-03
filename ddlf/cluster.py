# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
import inspect
from tensorflow.keras import datasets, layers, models, optimizers, utils
from typing import List
from ddlf.config import *
from ddlf.iworker import *
from ddlf.proxyworker import *
from ddlf.request import *
from ddlf.response import *
from ddlf.status import *
from ddlf.tools import *


class Cluster(IWorker):
    def __init__(self):
        self.hosts = config.hosts
        self.ports = config.ports
        self.N: int = len(self.hosts)
        self.workers: List[ProxyWorker] = []

    async def connect(self):
        for id, (host, port) in enumerate(zip(self.hosts, self.ports)):
            worker = ProxyWorker(id + 1, host, port)
            self.workers.append(worker)
            await worker.connect()

    async def add_method(self, method):
        method_code = inspect.getsource(method)
        res = await asyncio.gather(
            *[worker.add_method(method_code=method_code, method_name=method.__name__) for worker in self.workers],
            return_exceptions=True)
        return res

    async def clean(self):
        res = await asyncio.gather(*[worker.clean() for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def close(self):
        res = await asyncio.gather(*[worker.close() for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def load_cifar10(self):
        res = await asyncio.gather(*[worker.load_cifar10()
                                     for worker in self.workers], return_exceptions=True)
        return res

    async def load_mnist(self):
        res = await asyncio.gather(*[worker.load_mnist()
                                     for worker in self.workers], return_exceptions=True)
        return res

    async def load_partition(self, permutation):
        res = await asyncio.gather(*[worker.load_partition(permutation=permutation)
                                     for worker in self.workers], return_exceptions=True)
        return res

    async def ping(self):
        res = await asyncio.gather(*[worker.ping() for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def remove_method(self, method):
        res = await asyncio.gather(*[worker.remove_method(method.__name__) for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def run(self, method, **kwargs):
        res = await asyncio.gather(*[worker.run(method_name=method.__name__, **kwargs) for worker in self.workers],
                                       return_exceptions=True)
        return res

    async def run_on(self, worker: ProxyWorker, method, **kwargs):
        res = await worker.run(method_name=method.__name__, **kwargs)
        return res

    async def run_code(self, code):
        res = await asyncio.gather(*[worker.run_code(code=code) for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def run_method1(self, method, **kwargs):
        method_code = inspect.getsource(method)
        res = await asyncio.gather(
            *[worker.run_method(method_code=method_code, method_name=method.__name__, **kwargs) for worker in
              self.workers],
            return_exceptions=True)
        return res

    async def run_method(self, method, *args, **kwargs):
        method_code = inspect.getsource(method)
        N = len(args)
        # if args are provided then run the method with different args on each worker node
        if N > 0:
            res = await asyncio.gather(
                *[worker.run_method(method_code=method_code, method_name=method.__name__, **args[i])
                for i, worker in enumerate(self.workers[:N])], return_exceptions=True)
        # if kwargs are provided then run the method with the same kwargs on each worker node
        # elif len(kwargs)>0:
        else:
            res = await asyncio.gather(
                *[worker.run_method(method_code=method_code, method_name=method.__name__, **kwargs)
                for worker in self.workers], return_exceptions=True)
        return res

    async def run_method_on(self, worker: ProxyWorker, method, **kwargs):
        method_code = inspect.getsource(method)
        res = await worker.run_method(method_code=method_code, method_name=method.__name__, **kwargs)
        return res

    async def show_data(self):
        res = await asyncio.gather(*[worker.show_data() for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def shutdown(self):
        res = await asyncio.gather(*[worker.shutdown() for worker in self.workers],
                                   return_exceptions=True)
        return res

    async def train(self, weights, worker_epochs, batch_size):
        res = await asyncio.gather(*[worker.train(weights=weights,
                                                  worker_epochs=worker_epochs,
                                                  batch_size=batch_size) for worker in self.workers],
                                   return_exceptions=True)
        return res
