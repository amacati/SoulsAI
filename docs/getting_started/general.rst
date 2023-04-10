The General Idea
================

Training reinforcement learning agents takes time. Like, a lot of time. Depending on the speed at
which we can sample from our environment, it might take days or weeks to train a single agent. This
is problematic for multiple reasons: We can't iterate fast on our hyperparameters, we can't determine
early if our agent is learning anything, retraining after environment changes is a pain, etc.

All of this is especially true when training agents with Dark Souls as the environment. To alleviate
this problem, we can essentially take two approaches: Make the game run faster, or run multiple
instances at once and somehow share the experience.

Increasing the Game Speed
^^^^^^^^^^^^^^^^^^^^^^^^^
The game is usually restricted to running in real time. This means that if our agent requires about
100 hours of playtime to converge to a satisfactory policy, it will literally take 100+ hours of real
time to train. This is obviously not ideal. Fortunately, with a few tricks outlined
`here <https://soulsgym.readthedocs.io/en/latest/core/speedhack.html/>`_, we can increase the game
speed by a factor of 2-3, depending on the available hardware. This cuts training times significantly,
but ultimately does not scale well.

Running Multiple Instances
^^^^^^^^^^^^^^^^^^^^^^^^^^
The second approach is to run multiple instances of the game at once. This is a bit more involved,
since Steam does not allow us to run the same game multiple times at once. However, we can run the
game on several machines. ``soulsai`` is designed to facilitate this approach. It allows us to run
independent train clients on different machines, and a single server that aggregates the experience.
The communication between the clients and the server is handled by ``redis``. Depending on the training
algorithm, the training supports asynchronous client updates, is resilient to client crashes, and recovers
from disconnects. The server can also be run on a different machine than the clients, allowing us to
completely isolate the sampling of experience from the training of neural networks.

Using this approach, we can improve the training speed by increasing the number of clients. The architecture
is also more flexible, since we can easily add or remove clients as needed, even during existing training runs.
Our training is also be more resilient to client crashes, since we can simply restart the crashed client. This is
important for training on ``soulsgym`` environments, since stability is not guaranteed.

The limiting factor for this approach is either the number of clients we can run at once, or the
update speed of the train server if the sample rate exceeds the threshold where the sample count required
for an update is reached before the server was able to compute the previous update.

The ``soulsai`` package is designed to facilitate this approach, and enable members of the Dark Souls
community to jointly train agents on Dark Souls boss fights.