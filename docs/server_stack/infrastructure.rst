.. _infrastructure-nodes:

Infrastructure nodes
====================

If the server stack runs on an external machine, a few basic infrastructure functionalities are
required to make it reachable by its domain name, and to add SSL encryption. We use NGINX as reverse
proxy and certbot to renew the SSL certificates. Both services run in the reverse_proxy docker
compose environment. In addition, you can use ddclient to update the DNS records of your domain.

On the first start, you need to modify the NGINX configuration file and disable HTTPS. Certbot has
not yet installed the certificates, and NGINX wouldn't be able to use HTTPS. Furthermore, you need
to ensure that the NGINX container launches before certbot so that it can serve the challenge files.
Once the certificates are installed, you can enable HTTPS again and switch to the renewal command
for certbot. Both commands are provided in the docker-compose file, you just have to uncomment the
relevant lines.

If you run the training server on windows, you also need to make sure that you created a firewall
rule allowing the connection to port 80 and 443 for the monitoring stack, and 6379 for Redis.
Additionally, it is very convenient to use WSL2 for the server. If you opt for this solution, make
sure to forward the ports from WSL2 to Windows. You can find the required commands and accompanying
explanations `here <https://jwstanly.com/blog/article/Port+Forwarding+WSL+2+to+Your+LAN/>`_.

A detailed tutorial on how to configure NGINX and certbot can also be found
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

ddclient
^^^^^^^^
ddclient is a dynamic DNS client that can update the DNS records of a domain name. If you are running
a local server stack, you can use ddclient to make it reachable by its domain name. Be sure to modify
the ddclient configuration file to match your domain name and credentials. The username and password
are stored in the *ddclient/config/ddclient.secret* file (the same folder as
*ddclient/config/ddclient.conf*). The file is ignored by git, so you can safely store your
credentials there.