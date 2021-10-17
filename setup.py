from setuptools import setup

# Use README.md as long_description
long_description = open("README.md").read()

# Read requirements
requirements = [line.strip() for line in open("requirements.txt").readlines()]

setup(
    name="finfast",
    version="0.0.0.dev0",
    description="Fast financial analysis toolkit for CPU and GPU",
    url="https://github.com/snugfox/finfast",
    author="SnugFox",
    author_email="snugfox@users.noreply.github.com",
    license="MIT",
    python_requires=">=3.8.0",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["finfast", "finfast_torch"],
)
