# Contributing

There are multiple ways in which you can contribute:

**Report a bug**: To report a suspected bug, please raise an issue on Github. Please try to give a short but complete description of the issue. Use ```bug``` as a label on the Github issue.

**Suggest a feature**: To suggest a new feature, please raise an issue on Github. Please describe the feature and the intended use case. Use ```enhancement``` as a label on the Github issue.

## Developing NoisePy

NoisePy is going under major re-development. Part of the core development involves adding data objects and stores, modularizing it to facilitate community development, and giving alternative workflows for HPC, Cloud, and DAS.

Fork the repository, and create your local version, then follow the installation steps:
```bash
conda create -n noisepy python=3.8 pip
conda activate noisepy
conda install -c conda-forge openmpi
pip install -e ".[dev]"
```

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
