from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

setup(
    name="your-package-name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires
)