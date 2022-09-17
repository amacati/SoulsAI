class SoulsAIException(Exception):
    """Base class for SoulsAIExceptions."""



class InvalidConfigError(SoulsAIException):
    """Raised when a faulty configuration is detected."""


class MissingConfigError(SoulsAIException):
    """Raised when a required configuration file is missing."""