.. _server-overview:

Overview
========

The server module contains nodes for training the agents, live telemetry, and a web interface.

The whole stack is dockerized to maximize the isolation of nodes and to make it easier to manage the
setup.

.. note::
    It is entirely possible to train on a single machine by configuring the server to run in WSL2.
    To do so, simply launch the server containers and set the required addresses in the
    configuration files to ``localhost``.

The server is composed of a :ref:`core <core-nodes>` stack for communication, training and telemetry, a
:ref:`monitoring interface <monitoring>` and some :ref:`infrastructure nodes <infrastructure-nodes>`.