from setuptools import setup, find_namespace_packages

setup(
    name='modules',
    version='0.1.0',
    description='A brief description of my project',
    packages=find_namespace_packages(include=['pydevd_plugins.*']),
)