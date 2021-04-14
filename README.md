
Pytorch implementation for box embeddings and box representations.

<p align="center">
  <img src="/images/boxes.png">
</p>

## Status

![Tests](https://github.com/iesl/box-embeddings/workflows/Tests/badge.svg) ![Typing/Doc/Style](https://github.com/iesl/box-embeddings/workflows/Typing/Doc/Style/badge.svg)


## Contributors

1. Dhruvesh Patel [@dhruvdcoder](https://github.com/dhruvdcoder) (Active Maintainer)

2. Shib Sankar Dasgupta [@ssdasgupta](https://github.com/ssdasgupta) (Active Maintainer)

3. Michael Boratko [@mboratko](https://github.com/mboratko)

4. Xiang (Lorraine) Li [@Lorraine333](https://github.com/Lorraine333)

5. Trang Tran [@trangtran72](https://github.com/trangtran72)

## Installation

### Installing via pip

The preferred way to install Box Embeddings is via `pip`. Just run `pip install box-embeddings`

### Installing from source

You can also install Box Embeddings by cloning our git repository

```
git clone https://github.com/iesl/box-embeddings
```

Create a Python 3.7 or 3.8 virtual environment, and install Box Embeddings in editable mode by running:

```
pip install --editable . --user
pip install -r core_requirements.txt
```
## Package Overview
| Command | Description |
| --- | --- |
| `box_embeddings` | An open-source NLP research library, built on PyTorch & TensorFlow |
| `box_embeddings.common` | Utility modules that are used across the library |
| `box_embeddings.initializations` | Initialization modules |
| `box_embeddings.modules` | A collection of modules to operate on boxes|
| `box_embeddings.parameterizations` | A collection of modules to parameterize boxes|


## Citing

1. If you use simple hard boxes with surrogate loss then cite the following paper:

```
@inproceedings{vilnis2018probabilistic,
  title={Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures},
  author={Vilnis, Luke and Li, Xiang and Murty, Shikhar and McCallum, Andrew},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={263--272},
  year={2018}
}
```

2. If you use softboxes without any regularizaton the cite the following paper:

```
@inproceedings{
li2018smoothing,
title={Smoothing the Geometry of Probabilistic Box Embeddings},
author={Xiang Li and Luke Vilnis and Dongxu Zhang and Michael Boratko and Andrew McCallum},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=H1xSNiRcF7},
}
```

3. If you use softboxes with regularizations defined in the `Regularizations` module then cite the following paper:

```
@inproceedings{
patel2020representing,
title={Representing Joint Hierarchies with Box Embeddings},
author={Dhruvesh Patel and Shib Sankar Dasgupta and Michael Boratko and Xiang Li and Luke Vilnis and Andrew McCallum},
booktitle={Automated Knowledge Base Construction},
year={2020},
url={https://openreview.net/forum?id=J246NSqR_l}
}
```

The code for this library can be found [here](https://github.com/iesl/box-embeddings).

## Contributions
We welcome all contributions from the community to make Box Embeddings a better package.
If you're a first time contributor, we recommend you start by reading our
[CONTRIBUTING.md](https://github.com/iesl/box-embeddings/blob/main/.github/CONTRIBUTING.md) guide.

## Team
Box Embeddings is an open-source project developed by the research team from the
[Information Extraction and Synthesis Laboratory](http://www.iesl.cs.umass.edu/) at the
[College of Information and Computer Sciences (UMass Amherst)](https://www.cics.umass.edu/).
