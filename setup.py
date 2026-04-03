import setuptools

setup(
    name="hiera-transformer",
    version="0.1.4",
    author="Chaitanya Ryali, Daniel Bolya",
    url="https://github.com/facebookresearch/hiera",
    description="A fast, powerful, and simple hierarchical vision transformer",
    install_requires=["torch>=1.8.1", "timm>=0.4.12", "tqdm", "packaging"],
    packages=find_packages(exclude=("examples", "build")),
    license="Apache 2.0",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
