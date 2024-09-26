# Align-Anything Documentation

This folder contains the source code for building the Align-Anything documentation, which is intended to inform you about how to participate in the development of the Align-Anything documentation.

## Building the documentation locally

You only need two simple steps to deploy and render the Align-Anything documentation on your local webpage.

1. Install the depency

```bash
# suppose you are in the root dir of align-anything
cd ./docs
# make sure your python env is activated
pip install -r ./requirements.txt
```

2. Build the docs with Sphinx

> [Sphinx](https://www.sphinx-doc.org/en/master/index.html) is a powerful documentation building tool that allows for web page editing using only Markdown or reStructuredText.

To build the page locally with Sphinx, you only need to run the following under `align-anything/docs/` folder:

```bash
sphinx-autobuild source source/_build/html
```

The default port for deployment is `8000`. If the port has already beed used, you can switch it to another port, *i.e.,* `8080`, by passing the `--port` arguments:

```bash
sphinx-autobuild --port 8080 source source/_build/html
```

## Documentation tutorial

Sphinx supports both Markdown and reStructuredText, making it very easy to get started. We have compiled relevant reference resources here to facilitate community contributions.

| Resource             | Link                                                                                                          | Description                        |
|----------------------|---------------------------------------------------------------------------------------------------------------|------------------------------------|
| Sphinx               | [link](https://www.sphinx-doc.org/en/master/index.html)                                                               | Official documentation of Sphinx   |
| Sphinx Extensions    | [link](https://www.sphinx-doc.org/en/master/usage/extensions/index.html)                                              | The extensions bundled with Sphinx |
| Sphinx Design    | [link](https://sphinxdocs.ansys.com/version/stable/examples/sphinx-design.html)                                              | The tutorial for Sphinx design plug in |
| Sphinx Tutorial (EN) | [link](https://medium.com/@pratikdomadiya123/build-project-documentation-quickly-with-the-sphinx-python-2a9732b66594) | The EN tutorial for Sphinx         |
| Sphinx Tutorial (CN) | [link](https://studynotes.readthedocs.io/zh/main/struct/extend/ext-index.html)                                        | The CN tutorial for Sphinx         |
| reStructuredText Tutorial (EN) | [link](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)                                        | The EN tutorial for reStructuredText         |