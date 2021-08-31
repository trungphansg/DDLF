# IWorker interface
# @author: Trung Phan
# @created date: 2021-07-01
# @last modified date:
# @note:
class IWorker(object):
    async def add_method(self, method_code, method_name):
        raise NotImplemented

    async def clean(self):
        raise NotImplemented

    async def close(self):
        raise NotImplemented

    async def connect(self):
        raise NotImplemented

    async def load_cifar10(self):
        raise NotImplemented

    async def load_mnist(self):
        raise NotImplemented

    async def ping(self):
        raise NotImplemented

    async def remove_method(self, method_name):
        raise NotImplemented

    async def run(self, method_name, **kwargs):
        raise NotImplemented

    async def run_code(self, code):
        raise NotImplemented

    async def run_method(self, method_code, method_name, **kwargs):
        raise NotImplemented

    async def show_data(self):
        raise NotImplemented

    async def shutdown(self):
        raise NotImplemented