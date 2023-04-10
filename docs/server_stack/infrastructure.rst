.. _infrastructure-nodes:

Infrastructure nodes
====================

If the server stack runs on an external machine, a few basic infrastructure functionalities are
required to make it reachable by its domain name, and to add SSL encryption. We use NGINX as reverse
proxy and certbot to renew the SSL certificates. Both services run in the reverse_proxy docker
compose environment.

A detailed tutorial on how to configure NGINX and certbot can be found
`here <https://mindsers.blog/post/https-using-nginx-certbot-docker/>`_.


Reverse Proxy
^^^^^^^^^^^^^
The NGINX container requires three volumes to be mapped correctly:

    * *nginx/conf*
    * *certbot/www*
    * *certbot/conf*

These paths have to point to the locations of the NGINX config folder, the certbot challenge files,
and the */etc/letsencrypt* folder, respectively.

Certbot
^^^^^^^
Certbot makes sure the SSL certificates are continuously updated. For more information about
Certbot, please have a look at `the official website <_https://certbot.eff.org/>`_.