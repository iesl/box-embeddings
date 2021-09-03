<h1 align="center">Box Embeddings</h1>

A fully open source Python library for **geometric representation learning**, compatible with both PyTorch and
TensorFlow, which allows existing neural network layers to be replaced  with or transformed into boxes easily.

![Tests](https://github.com/iesl/box-embeddings/workflows/Tests/badge.svg)
![Typing/Doc/Style](https://github.com/iesl/box-embeddings/workflows/Typing/Doc/Style/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iesl/box-embeddings/dev/main)
[![codecov](https://codecov.io/gh/iesl/box-embeddings/branch/main/graph/badge.svg?token=XPNQI0QXFZ)](https://codecov.io/gh/iesl/box-embeddings)

<hr/>

## 🌟 Features
- Modular and reusable library that aids the researchers in studying probabilistic box embeddings.
- Extensive documentation and example code, demonstrating the use of the library to make it easy to
  adapt to existing codebases.
- Rigorously unit-test the codebase with high coverage, ensuring an additional layer of reliability.
- Customizable pipelines
- Actively being maintained by [IESL at UMass](http://www.iesl.cs.umass.edu/)

## 💻 Installation

### Installing via pip

The preferred way to install Box Embeddings for regular usage, test, or integration into the existing workflow
is via `pip`. Just run

`pip install box-embeddings`

### Installing from source

You can also install Box Embeddings by cloning our git repository

```
git clone https://github.com/iesl/box-embeddings
```

Create a Python 3.7 or 3.8 virtual environment under the project directory and install the `Box Embeddings`
module in editable mode by running:

```shell
virtualenv box_venv
source box_venv/bin/activate
pip install --editable . --user
pip install -r core_requirements.txt
```
## 👟 Quick start
After installing `Box Embeddings`, a box can be initialized from a tensor as follows:

```python
import torch
from box_embeddings.parameterizations.box_tensor import BoxTensor
data_x = torch.tensor([[1,2],[-1,5]])
box_1 = BoxTensor(data_x)
box_1
```

The result `box_1` is now a `BoxTensor` object. To view other examples, visit the
[examples section](https://github.com/iesl/box-embeddings/tree/main/usage_doc).

```python
BoxTensor(tensor([[ 1,  2],
        [-1,  5]]))
```


## 📖 Command Overview
| Command | Description |
| --- | --- |
| `box_embeddings` | An open-source library for NLP or graph learning |
| `box_embeddings.common` | Utility modules that are used across the library |
| `box_embeddings.initializations` | Initialization modules |
| `box_embeddings.modules` | A collection of modules to operate on boxes|
| `box_embeddings.parameterizations` | A collection of modules to parameterize boxes|


## 📚 Reference

1. If you use simple hard boxes with surrogate loss then cite the following paper:

```bibtex
@inproceedings{vilnis2018probabilistic,
  title={Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures},
  author={Vilnis, Luke and Li, Xiang and Murty, Shikhar and McCallum, Andrew},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for
  Computational Linguistics (Volume 1: Long Papers)},
  pages={263--272},
  year={2018}
}
```

2. If you use softboxes without any regularizaton the cite the following paper:

```bibtex
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

```bibtex
@inproceedings{
patel2020representing,
title={Representing Joint Hierarchies with Box Embeddings},
author={Dhruvesh Patel and Shib Sankar Dasgupta and Michael Boratko and Xiang Li and Luke Vilnis
and Andrew McCallum},
booktitle={Automated Knowledge Base Construction},
year={2020},
url={https://openreview.net/forum?id=J246NSqR_l}
}
```
4. If you use Gumbel box then cite the following paper:

```bibtex
@article{dasgupta2020improving,
  title={Improving Local Identifiability in Probabilistic Box Embeddings},
  author={Dasgupta, Shib Sankar and Boratko, Michael and Zhang, Dongxu and Vilnis, Luke
  and Li, Xiang Lorraine and McCallum, Andrew},
  journal={arXiv preprint arXiv:2010.04831},
  year={2020}
}
```

## 💪 Contributors

* Dhruvesh Patel [@dhruvdcoder](https://github.com/dhruvdcoder)

* Shib Sankar Dasgupta [@ssdasgupta](https://github.com/ssdasgupta)

* Michael Boratko [@mboratko](https://github.com/mboratko)

* Xiang (Lorraine) Li [@Lorraine333](https://github.com/Lorraine333)

* Trang Tran [@trangtran72](https://github.com/trangtran72)

* Purujit Goyal [@purujitgoyal](https://github.com/purujitgoyal)

* Tejas Chheda [@tejas4888](https://github.com/tejas4888)

We welcome all contributions from the community to make Box Embeddings a better package.
If you're a first time contributor, we recommend you start by reading our
[CONTRIBUTING.md](https://github.com/iesl/box-embeddings/blob/main/.github/CONTRIBUTING.md) guide.

## 🤗 Acknowledgments
Box Embeddings is an open-source project developed by the research team from the
[Information Extraction and Synthesis Laboratory](http://www.iesl.cs.umass.edu/) at the
[College of Information and Computer Sciences (UMass Amherst)](https://www.cics.umass.edu/).
