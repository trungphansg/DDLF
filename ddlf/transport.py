# Transport functions
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date: 2021-07-16
# @note: using pickle for serialization

import pickle


async def send_message(message, writer):
    msg = pickle.dumps(message)
    writer.write(len(msg).to_bytes(4, byteorder='big'))
    writer.write(msg)
    await writer.drain()


async def recv_message(reader):
    data = await reader.readexactly(4)
    msglen = int.from_bytes(data, byteorder='big')
    msg = await reader.readexactly(msglen)
    message = pickle.loads(msg)
    return message
