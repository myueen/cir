from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()

setup(
    name='contrastive_inverse_regression',
    version='1.0.3',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.1.4',
        'scipy>=1.9.3'
    ],

    long_description=description,
    long_description_content_type="text/markdown",
)
