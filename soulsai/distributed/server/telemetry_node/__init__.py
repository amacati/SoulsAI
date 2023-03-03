"""The telemetry node module tracks the current agent performance on clients during training.

The node is supposed to run within a Docker container. If live monitoring is enabled, the
``monitoring_net`` Docker network has to exist to communicate with Grafana.
"""
