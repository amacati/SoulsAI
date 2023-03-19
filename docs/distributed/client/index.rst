distributed.client
==================
Each algorithm has its own custom loop, since requirements differ from algorithm to algorithm.
However, all clients share the abstraction of a connector to communicate with the ``Redis`` server.

.. toctree::
    :hidden:

    main
    dqn_client
    ppo_client
    connector
    watchdog

Clients
~~~~~~~
The clients are responsible for sampling from the gym environment. Samples are sent to a central
training server using a ``Connector`` and the ``Redis`` messaging service.

Connector
~~~~~~~~~
The :ref:`distributed.client.connector` classes abstract the communication with Redis. They are
responsible for establishing the connection, sending samples and telemetry, and error handling in
case the client disconnects. Since we want to sample as fast as possible from the game, clients only
push messages to the connector into multiprocessed queues, and samples are uploaded to the server in
a separate process. In addition, they download available network updates and make them available in
the client process.

Watchdog
~~~~~~~~
To improve the reliablity of the training process, the training process can also be handled by a
:class:`Client Watchdog <soulsai.distributed.client.watchdog.ClientWatchdog>`. This watchdog
regularly checks if the sampling rate of the client is still within expected values, and restarts
the training if not.
