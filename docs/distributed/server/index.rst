distributed.server
==================

.. toctree::
    :hidden:

    training_node/index
    telemetry_node/index

The server module contains nodes for training the agents, live telemetry, and a web interface.

The whole stack is dockerized to maximize the isolation of nodes and to make it easier to manage the
setup.

.. note::
    It is entirely possible to train on a single machine by configuring the server to run in WSL2.
    To do so, simply launch the server containers and set the required addresses in the
    configuration files to ``localhost``.

You can find a more detailed description of the server stack in the 