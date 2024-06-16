from setuptools import setup, find_packages

setup(
    name='contrastive_inverse_regression',
    version='1.0.4',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=2.0.1',
        'scipy>=1.9.3'
    ],

    long_description=description,
    long_description_content_type="text/markdown",
    url='https://github.com/myueen/contrastive-inverse-regression',
    author='Yueen Ma',
    author_email='myueen@ad.unc.edu',
    license='MIT',
    python_requires='>=3.10',
)
