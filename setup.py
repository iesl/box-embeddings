from typing import List
from setuptools import setup, find_packages
import os

PATH_ROOT = os.path.dirname(__file__)
with open("README.md", "r") as fh:
    long_description = fh.read()


def load_requirements(
    path_dir: str = PATH_ROOT, comment_char: str = "#"
) -> List:
    with open(os.path.join(path_dir, "core_requirements.txt"), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []

    for ln in lines:
        # filer all comments

        if comment_char in ln:
            ln = ln[: ln.index(comment_char)]

        if ln:  # if requirement is not empty
            reqs.append(ln)

    return reqs


install_requires = load_requirements()

setup(
    name="box_embeddings",
    version="0.1.0",
    author="Dhruvesh Patel, Shib Sankar Dasgupta, Michael Boratko, Purujit Goyal, Tejas Chheda, Trang Tran, Xiang (Lorraine) Li",
    author_email="1793dnp@gmail.com",
    description="Pytorch and Tensorflow implemention of box embedding models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.iesl.cs.umass.edu/box-embeddings/",
    project_urls={
        "Documentation": "http://www.iesl.cs.umass.edu/box-embeddings",
        "Source Code": "https://github.com/iesl/box-embeddings",
    },
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "examples"]
    ),
    package_data={"box_embeddings": ["py.typed"]},
    install_requires=install_requires,
    keywords=[
        "pytorch",
        "tensorflow",
        "AI",
        "ML",
        "Machine Learning",
        "Deep Learning",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.5",
)
