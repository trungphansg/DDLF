# Distributed DL Training Framework

## Information
@author: Trung Phan \
@email: trungphansg@gmail.com \
@created date: 2021-06-28 \
@last modified date: 2021-08-22\
@version: 2.2

## Installation
* python3.7+ 
* pip install numpy pympler tensorflow pyyaml

* Master node:
    - worker.py
    - request.py
    - tools.py
    - transport.py
* Worker nodes:
    - config.py
    - rpc.py
    - proxyworker.py
    - app.py
    - task-*.py
    
## Architecture
<img src="images/architecture.png" width="50%">

## Files
* worker: runs on worker nodes
* config, proxy, master, app, tasks: runs on the master nodes

## Configuration
* Open file config.py:
    * hosts: list of worker nodes
    * ports: list of worker nodes' ports

## Running
* cd ddl 
* start N (N is the number of workers): starting the system
* python3.8 task-*: running a task
* shutdown: shutting down the system

## Testing
* start-cluster.py: start the cluster
* task-*.py: execute a task

## References
https://stackabuse.com/python-async-await-tutorial \
https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently \
https://docs.python.org/3/library/asyncio-stream.html \
https://stackoverflow.com/questions/62383366/asyncio-streamwriter-sending-multiple-writes