"""Setup file for the SoulsGym module."""
from setuptools import setup, find_packages

setup(
    name="soulsai",
    packages=find_packages(),
    version="0.1",
    description="Learning algorithm collection for SoulsGym and DarkSouls III",
    author=["Martin Schuck", "Raphael Krauthann"],
    author_email="real.amacati@gmail.com",
    url="https://github.com/amacati/SoulsAI",
    # TODO: FILL IN CORRECT ARCHIVE
    download_url="https://github.com/amacati/SoulsAI/archive/v_01.tar.gz",
    keywords=["Reinforcement Learning", "gym", "Dark Souls"],
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha", "Intended Audience :: Developers",
        "Intended Audience :: Education", "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", "Programming Language :: Python :: 3.9"
    ])
