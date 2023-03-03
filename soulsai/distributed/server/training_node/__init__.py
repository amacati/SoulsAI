"""The training node module contains the learning algorithms for training the agents.

Training nodes receive samples from client nodes, train the networks, and broadcast the updated
agents to all clients either in a synchronous or asynchronous manner. The Dockerfile allows to run
the node in a container and automatically launches the correct algorithm depending on the training
configuration.
"""
