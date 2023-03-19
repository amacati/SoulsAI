.. SoulsAI documentation master file, created by
   sphinx-quickstart on Sun Feb 12 13:48:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SoulsAI documentation
=====================

The SoulsAI package implements distributed reinforcement learning algorithms to train agents on
``SoulsGym`` environments.

Results
~~~~~~~
The video below shows a reinforcement learning agent beating Iudex Gundyr after training on
approximately 5 million environment samples. The game runs at 3x speed to accelerate training. The
agent is trained using duelling advantage networks with multi-step returns.

.. image:: img/iudex_speedup_1.gif
   :alt: Reinforcement learning agent fights against Iudex Gundyr in Dark Souls III.
   :align: center

.. raw:: html

    <div style="margin-bottom: 2em;"></div>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/setup
   getting_started/starting_to_train

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Server Stack

   server_stack/overview
   server_stack/core
   server_stack/monitoring
   server_stack/infrastructure

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Python API

   soulsai

   core/index

   distributed/index

   data/index

   utils/index

   exception/exception


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
