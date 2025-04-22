from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rl_benchmark",
    version="0.1.0",
    author="Valentin Macé",
    author_email="valentin.mace@kedgebs.com",
    description="A benchmark suite for reinforcement learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valentinmace/rl_benchmark",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
