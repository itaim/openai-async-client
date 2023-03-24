from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


def read_version():
    with open("VERSION", "r", encoding="utf-8") as fh:
        return fh.read().strip()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openai-async-client",
    version=read_version(),
    author="Itai Marks",
    author_email="itai.marks@gmail.com",
    description="OpenAI async API with client side timeout, retry with exponential backoff and connection reuse",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/itaim/openai-async-client.git",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7.16",
    install_requires=requirements,
)
