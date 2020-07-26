
Pytorch implementation for box embeddings as well as box representations.

![Boxes Picture](/images/boxes.png)


# Contributors

1. Dhruvesh Patel [@dhruvdcoder](https://github.com/dhruvdcoder) (Active Maintainer)

2. Shib Shankar Dasgupta [@ssdasgupta](https://github.com/ssdasgupta) (Active Maintainer)

3. Michael Boratko [@mboratko](https://github.com/mboratko)

4. Xiang (Lorraine) Li [@Lorraine333](https://github.com/Lorraine333)

# Cite

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

