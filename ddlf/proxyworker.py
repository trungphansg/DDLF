# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date: 2021-07-18
# @note: adding id to ProxyWorker
import asyncio
import pickle
from ddltrain.iworker import *
from ddltrain.request import *
from ddltrain.transport import *


class ProxyWorker(IWorker):
    def __init__(self, id, host, port):
        self.id = id
        self.host = host
        self.port = port

    def __str__(self):
        return f'Proxy worker: {{ip:{self.host}, port: {self.port}}}'

    async def connect(self):
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            return self.host, 'OK'
        except Exception as e:
            return self.host, e

    async def rpc(self, req):
        # send the request
        await send_message(req, self.writer)
        # receive the result
        return await recv_message(self.reader)

    async def add_method(self, method_code, method_name):
        return await self.rpc(Request('add_method', method_code, method_name))

    async def clean(self):
        return await self.rpc(Request('clean'))

    async def close(self):
        res = await self.rpc(Request('close'))
        # close the connection
        self.writer.close()
        return res

    async def load_cifar10(self):
        return await self.rpc(Request('load_cifar10'))

    async def load_mnist(self):
        return await self.rpc(Request('load_mnist'))

    async def ping(self):
        return await self.rpc(Request('ping'))

    async def remove_method(self, method_name):
        return await self.rpc(Request('remove_method', method_name))

    async def run(self, method_name, **kwargs):
        return await self.rpc(Request('run', method_name, **kwargs))

    async def run_code(self, code):
        return await self.rpc(Request('run_code', code))

    async def run_method(self, method_code, method_name, **kwargs):
        return await self.rpc(Request('run_method', method_code, method_name, **kwargs))

    async def show_data(self):
        return await self.rpc(Request('show_data'))

    async def shutdown(self):
        res = await self.rpc(Request('shutdown'))
        # close the connection
        self.writer.close()
        return res


