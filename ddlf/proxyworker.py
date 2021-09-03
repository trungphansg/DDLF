# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date: 2021-07-18
# @note: adding id to ProxyWorker
import asyncio
import sys
from ddlf.iworker import *
from ddlf.request import *
from ddlf.response import *
from ddlf.status import *
from ddlf.transport import *


class ProxyWorker(IWorker):
    def __init__(self, id, host, port):
        self.id = id
        self.host = host
        self.port = port

    def __str__(self):
        return f'Proxy worker: {{ip:{self.host}, port: {self.port}}}'

    async def connect(self):
        status = Status.OK
        result = None
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        except Exception as e:
            status = Status.ERROR
            result = e
        self.show_status(command='connect', status=status, result= result)

    async def rpc(self, req):
        # send the request
        await send_message(req, self.writer)
        # receive the result
        res = await recv_message(self.reader)
        # notification
        self.show_status(command=req.command, status=res.status, result=res.result)
        return res.result

    def show_status(self, command, status, result):
        print(f"* {command.upper()} {'-' * (30 - len(command))}")
        print(f"From host: {self.host}:{self.port}")
        print(f"Status: {status}")
        if status == Status.ERROR:
            print(f"Exception: {result}")
            sys.exit()

    async def add_method(self, **kwargs):
        return await self.rpc(Request('add_method', **kwargs))

    async def clean(self):
        return await self.rpc(Request('clean'))

    async def close(self):
        response = await self.rpc(Request('close'))
        # close the connection
        self.writer.close()
        return response

    async def load_cifar10(self):
        return await self.rpc(Request('load_cifar10'))

    async def load_mnist(self):
        return await self.rpc(Request('load_mnist'))

    async def load_partition(self, **kwargs):
        return await self.rpc(Request('load_partition', **kwargs))

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
        response = await self.rpc(Request('shutdown'))
        # close the connection
        self.writer.close()
        return response

    async def train(self, **kwargs):
        return await self.rpc(Request('train', **kwargs))


