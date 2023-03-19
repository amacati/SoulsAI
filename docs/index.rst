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

.. raw:: html

   <div style="position: relative; padding-bottom: 2em; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="720" height="405" src="https://www.youtube.com/embed/bfiS6bzOLiE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>

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
