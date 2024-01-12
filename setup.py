from setuptools import setup, find_packages

setup(
    name='cir',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.1.4',
        'scipy>=1.9.3'
    ]
)
