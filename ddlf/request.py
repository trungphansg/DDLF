# Representing requests for RPC
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date: 2021-06-28
# @note:
class Request(object):

    def __init__(self, command, *args, **kwargs):
        # create model
        self.command = command # command name
        self.args = args # Non Keyword Arguments
        self.kwargs = kwargs # Keyword Arguments

    def __str__(self):
        return f'request = {{command: {self.command}, args: {self.args}, kwargs: {self.kwargs}}}'
