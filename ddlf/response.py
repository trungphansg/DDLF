# Representing responses for RPC
# @author: Trung Phan
# @created date: 2021-09-01
# @last modified date:
# @note:
class Response(object):

    def __init__(self, status, result):
        self.status = status # status of RPC
        self.result = result # result of RPC

    def __str__(self):
        return f"status={self.status}, result={self.result}"
