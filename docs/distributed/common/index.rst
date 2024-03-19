distributed.common
==================

The distributed common module contains the serialization and deserialization functions shared by the
clients and servers. We use ``TensorDict``s and leverage the ``torch.save`` and ``torch.load``
functions to avoid writing custom serialization code. As a consequence, any data that has to be sent
over the network must be expressed as a ``Tensor`` or a ``TensorDict``.

.. toctree::
    :hidden:

    serialization
