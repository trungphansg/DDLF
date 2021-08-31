# Distributed DL Client runs on the master node
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
from ddlf.cluster import *

code1 = '''
    def f(self):
        self.data['total'] = 1000
        
    setattr(Worker, 'ff', f)    
    self.ff()
    '''

code2 = '''
    self.data['price'] = 1000000
    '''

async def g(self, a, b):
    self.data['unit_price'] = 10000 + a - b
    return self.data['unit_price']


async def main():
    cluster = Cluster()
    await cluster.connect()
    await cluster.run_code(code1)
    await cluster.run_code(code2)
    res = await cluster.run_method(g, a=4000, b=2000 )
    print(f"result: {res}")
    await cluster.show_data()
    # await cluster.clean()
    await cluster.close()


asyncio.run(main())
