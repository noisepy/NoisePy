# Contributing

After cloning the repo and creating a virtual environment with either **pip** o **conda**:

Do an editable installation to get the dependencies (from the project root):
```sh
$ pip install -e ".[dev]"


Install the `pre-commit` hook:
```sh
$ pre-commit install
```

This will run the linting and formatting checks configured in the project before every commit.

## Using VS Code

The following extensions are recommended:

- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
- [black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)
