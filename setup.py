from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="rolebotics.io",
    version="0.1.0",
    python_requires=">=3.7.16, <4",
    packages=find_packages(),
    install_requires=requirements,
    description="OpenAI async client",
    author="itai.marks@gmail.com",
    url="https://github.com/itaim/rolebotics.io",
    license="MIT",
)
