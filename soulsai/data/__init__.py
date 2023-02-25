"""The data module provides the means to transform environment observations into numeric arrays.

The :class:`.GameStateTransformer` transforms ``GameState`` objects into numeric arrays. In
addition, it allows to bin common animations into a single categorical encoding and combines the
animation duration.

Animations are converted into one-hot encodings by passing them to a :class:`.OneHotEncoder`. The
encoder is provided to avoid the dependency on additional packages such as ``sklearn``.
"""
