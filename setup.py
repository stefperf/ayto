from setuptools import setup, find_packages

setup(
    name='ayto',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.3',
    ],
    python_requires='>=3.12',
)
