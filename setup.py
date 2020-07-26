from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["torch>=1.3.0"]

setup(
    name='box_embeddings',
    version='0.0.1',
    author="Dhruvesh Patel",
    author_email="1793dnp@gmail.com",
    description='Pytorch implemention of box embedding models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.iesl.cs.umass.edu/box-embeddings/",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "examples"]),
    package_data={'box_embeddings': ['py.typed']},
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", 'Development Status :: 3 - Alpha'
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.5')
