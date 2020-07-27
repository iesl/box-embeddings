# Setup

1. In order to contribute you will need to clone/fork the repo and install the test requirements locally.

```
pip install -r test_requirements.txt
pip install -r doc_requirements.txt
```

2. Install [pre-commit](https://pre-commit.com/).

```
pip install pre-commit
pre-commit install
```

3. After making changes, run tests locally.


## Code Styling

We use `flake8` and `black` for code formatting and `darglint` for checking doc-strings.
