# Distributed DL Server runs on worker nodes
# @author: Trung Phan
# @created date: 2021-06-28
# @last modified date:
# @note:
def subtract(array_list1: list, array_list2: list):
    return [x - y for x, y in zip(array_list1, array_list2)]

def divide(array_list: list, n: int):
    return [x / n for x in array_list]