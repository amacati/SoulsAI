.. _core-nodes:

Core nodes
==========

The core of the training server stack consists of the communication service node ``redis``,
the ``training_node`` and the ``telemetry_node``. Redis allows us to communicate with clients, via
messages and the data base. The training node actually trains the agents, and the telemetry node
aggregates the results to visualize the current training progress. These components are essential
for training

.. note::
    It is highly recommended to launch the server from its docker-compose file at the root of the
    repository! This ensures the proper mapping of volumes, networks, ports etc.

Redis
^^^^^
Redis runs in its own Docker container and is preconfigured to run as communication service for
external client nodes. 

.. note::
    To restrict access to Redis, both the server and the client have to add a *redis.secret* file in
    the config folder. This file should contain a single line reading

    .. code-block:: rst

        requirepass <xxx>
    
    where *<xxx>* is your Redis password. Please choose a password of sufficient length and make
    sure the password on your clients matches that of your server!

Training
^^^^^^^^
The main training node connects to Redis and receives the sample messages from clients. It then
updates the agent and broadcasts it to connected clients. Currently, the training nodes for DQN and
PPO are implemented.

Configuration
-------------
The training node is configured using the files in the config folder. By default, the 
*config_d.yaml* file contains all necessary arguments to run the server and clients. If you want to
change the training settings, you can add a *config.yaml* file to the config folder. Each entry in
this file will overwrite the respective default settings.

Checkpoints
-----------
On each training start, a unique folder is generated under the saves folder. It contains the
training configuration saved as a json file, the current node checkpoint, and the training
results for each episode. Every *n*-th iteration, the training node checkpoints its networks,
transforms, replay buffers etc. to the experiment results folder.

During training, an additional subfolder for the best model is created. If the telemetry node
identifies the current model iteration as the best one yet, it sends a quicksave command to the
training node. This current best model is saved to the best model subfolder.

Finally, users may manually checkpoint the training node by running the *quicksave.py* script in the
script folder.

Telemetry
^^^^^^^^^
The telemetry node aggregates the training statistics and saves an updated figure of the results
every other episode. It also tracks the best average reward and sends checkpoint messages to the
training node when a better model has been found.

Grafana Connection
------------------
While the training figures are sufficient for a retrospective analysis, it can also be very helpful
to have live telemetry. This is particularly true due to the long training times of SoulsGym
environments. SoulsAI enables this via a Grafana instance. To display the current training
performance in Grafana, the telemetry node contains an adapter that acts as a Grafana data source.

Docker networks
^^^^^^^^^^^^^^^
All three core nodes communicate over the *train_server_net* Docker network. Clients can only
communicate with the nodes inside the network over Redis. To this end, port 6379 of the network is
exposed. You can create the network by running

.. code-block:: console

    $ docker network create train_server_net


Running the Stack
^^^^^^^^^^^^^^^^^
To run the stack, simply launch the docker compose file at the root of the repository. This will start
the core containers and expose the Redis port to the host machine.

.. code-block:: console

    $ docker compose up