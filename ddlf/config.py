# You should modify this file to match your cluster
class config(object):
    # List of worker node names
    hosts = ['localhost' for i in range(3)]
    # List of ports on which worker nodes listen for connection requests from the master node.
    # These ports may be the same or different.
    ports = [8888 + i for i in range(3)]
