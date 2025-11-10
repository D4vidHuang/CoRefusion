from setuptools import setup, find_packages

setup(
    name="corefusion",
    version="0.1.0",
    author="Yongcheng Huang",
    author_email="your.email@example.com",
    description="Diffusion LLM-based Code Refactoring Localization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/D4vidHuang/CoRefusion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)
