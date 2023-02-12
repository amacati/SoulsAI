class SoulsAIException(Exception):
    """Base class for SoulsAIExceptions."""


class InvalidConfigError(SoulsAIException):
    """Raised when a faulty configuration is detected."""


class MissingConfigError(SoulsAIException):
    """Raised when a required configuration file is missing."""


class ClientRegistrationError(SoulsAIException):
    """Raised when a client can't connect to the train server."""


class ServerDiscoveryTimeout(SoulsAIException):
    """Raised when the server can't successfully complete the discovery phase."""


class ServerTimeoutError(SoulsAIException):
    """Raised when a response from the server takes too long."""
