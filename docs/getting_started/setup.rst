Setting up training
===================
If you are interested in training your own agents, you will need to set up both the client and a training server.
This can be done on your local machine or on a remote server. We recommend using a remote server to easily allow other clients to connect to your training runs.

Setting Up a Client
-------------------
Setting up a client is very simple. After downloading the repository, you can create an environment with all the dependencies from the ``environment.yaml`` file at the repository root. Please note that you will need to install ``soulsgym`` from source for now!
To configure your clients, please change the ``redis_address`` in you client configuration file to point to your training server, and save your Redis password in a new ``redis.secret`` file! This file should be located in the ``config`` folder and should contain a single line
``requirepass xxx``, where xxx is your password. The rest of the configuration is going to be loaded from the server. After completing the configuration, you can run the client by executing the ``main.py`` file in the ``client`` folder.

.. code-block:: console

    $ python soulsai\distributed\client\main.py

.. warning::

    You can only run the clients on Windows machines. Linux and Mac are not supported at the moment. This is due to the fact that the game hacks are specificly designed for Windows.

Running a Training Server
-------------------------
Running the training server is more involved. It is highly recommended you read through the :ref:`server documentation <server-overview>` before proceeding.
After you configured your machine, you can run the training server by launching the docker-compose file at the root of the repository.

.. note::

    Please make sure you have configured your training hyperparameters before launching the server. These are located in the ``config_d.yaml`` and ``config.yaml`` file within
    the config folder. In addition, you need to make sure that the ``redis.secret`` file is present in the ``config`` folder, and that its password matches your clients' password.

.. note::

    If you are running the training server on a remote machine, you will need to make sure that the Redis port is open and accessible from your client machines. If you also want to enable monitoring, you have to configure the port for Grafana as well.

.. code-block:: console

    $ docker compose up

This will start the training server. Afterwards, you should be able to run a training client on your machine and connect to the server.

.. note::

    Remember to configure the ``redis_address`` and the password in you client configuration file to point to your training server!
