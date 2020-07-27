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

# Workflow
1. Create and checkout new feature branch for your work.

1. Make changes, add awesome code!

2. Build docs locally and inspect the new html pages.

```
cd docs_source
make html
```

Open `docs/index.html` and see if the docs are rendered correctly.

3. Stage docs

```
git add docs
```

4. Run code-style requirement checks using pre-commit.

```
pre-commit run --all-files -v
```

Fix all documentation and typing issues shown in the output.

5. Git re-stage the files after fixing the errors and commit. When you commit, `pre-commit` will run once again but only on the changed files

6. Push to remote. Create a PR if your feature is completed.


## Code Styling

We use `flake8` and `black` for code formatting and `darglint` for checking doc-strings.
