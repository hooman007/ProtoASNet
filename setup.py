"""Install package."""
from setuptools import setup, find_packages

setup(
    name="Aortic_Stenosis_XAI",
    version="0.0.1",
    description=("Interpretable detection of Aortic Stenosis Severity"),
    long_description=open("README.md").read(),
    url="https://github.com/hooman007/ProtoASNet",
    install_requires=["numpy"],
    packages=find_packages("."),
    zip_safe=False,
)
