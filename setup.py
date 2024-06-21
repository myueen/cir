from setuptools import setup, find_packages

with open("README.md", "r") as f:
    description = f.read()


setup(
    name='contrastive_inverse_regression',
    version='1.0.9',
    packages=find_packages(),
    # install_requires=[
    #     'numpy>=1.0.3',
    #     'pandas>=1.0.1',
    #     'scipy>=1.1.3'
    # ],

    long_description=description,
    long_description_content_type="text/markdown",
    url='https://github.com/myueen/contrastive-inverse-regression',
    author='Yueen Ma',
    author_email='myueen@ad.unc.edu',
    license='MIT',
    python_requires='>=3.7',
)
