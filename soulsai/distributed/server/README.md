# Redis server setup with WSL2

## Check WSL2 IP

**In WSL2**, check the IP with 

```console
user@pc:~$ ifconfig
```

>**WARNING:** Do **NOT** use the WSL2 IP from Windows cmd ipconfig!

## Set up port forwarding in Windows
Open cmd with admin privileges. Forward port 6379 to WSL2 by executing

```console
C:\WINDOWS\system32> netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=6379 connectaddress=<WSL2_IP> connectport=6379
```
where <WSL2_IP> is your WSL2 IP address.

## Open up Windows firewall
Open the Windows firewall panel and create a new incoming rule for port 6379, allowing all connections to this port.