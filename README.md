[![PEP8 Check](https://github.com/amacati/SoulsAI/actions/workflows/github-actions.yaml/badge.svg)](https://github.com/amacati/SoulsAI/actions/workflows/github-actions.yaml)   [![Documentation Status](https://readthedocs.org/projects/soulsai/badge/?version=latest)](https://soulsai.readthedocs.io/en/latest/?badge=latest)


# Running the training server stack

Create the server Docker network with

>:~/ $ docker network create server

The base services are started from the root folder with

>:~/SoulsAI$ docker compose up

If you want to run the Grafana/Prometheus monitoring stack, launch the monitoring compose file

>:~/SoulsAI/soulsai/distributed/server/monitoring$ docker compose up

## Redis server setup with WSL2

### Check WSL2 IP

**In WSL2**, check the IP with 

```console
user@pc:~$ ifconfig
```

>**WARNING:** Do **NOT** use the WSL2 IP from Windows cmd ipconfig!

### Set up port forwarding in Windows
Open cmd with admin privileges. Forward port 6379 to WSL2 by executing

```console
C:\WINDOWS\system32> netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=6379 connectaddress=<WSL2_IP> connectport=6379
```
where <WSL2_IP> is your WSL2 IP address.

### Open up Windows firewall
Open the Windows firewall panel and create a new incoming rule for port 6379, allowing all connections to this port.