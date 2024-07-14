## Contributing to align-anything

If you are interested in contributing to align-anything, your contributions will fall into two categories:

1. You want to propose a new Feature and implement it
    - Create an issue about your intended feature, and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: <https://github.com/PKU-Alignment/align-anything/issues>.
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to <https://github.com/PKU-Alignment/align-anything>.

If you are not familiar with creating a Pull Request, here are some guides:

- <http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request>
- <https://help.github.com/articles/creating-a-pull-request/>

## Developing align-anything

To develop align-anything on your machine, here are some tips:

1. Clone a copy of align-anything from GitHub:

```bash
git clone https://github.com/PKU-Alignment/align-anything
cd align-anything/
```

2. Install align-anything in develop mode, with support for building the docs and running tests:

```bash
pip install -e .
```

## Codestyle

We are using [black codestyle](https://github.com/psf/black) (max line length of 100 characters) together with [isort](https://github.com/timothycrosley/isort) to sort the imports.

**Please run `make format`** to reformat your code. You can check the codestyle using `make lint`.

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html), them using the following template, which is similar to the [PyTorch docs style]:

Similar to the standard PyTorch Style docstring formatting rules, the following guidelines should be followed for docstring types (docstring types are the type information contained in the round brackets after the variable name):

- If Python Version is less than `3.10`, you need to add `from __future__ import annotations`.

- The `Callable`, `Any`, `Iterable`, `Iterator`, `Generator` types should have their first letter capitalized.

- The `list` and `tuple` types should be completely lowercase.

- Types should not be made plural. For example: `tuple of int` should be used instead of `tuple of ints`.

- The only acceptable delimiter words for types are `or` and `of`. No other non-type words should be used other than `optional`.

- The word `optional` should only be used after the types, and it is only used if the user does not have to specify a value for the variable. Default values are listed after the variable description. Example:

```python
my_var (int, optional): Variable description. Default: 1
```

- Basic Python types should match their type name so that the [Intersphinx](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) extension can correctly identify them. For example:
  - Use `str` instead of `string`.
  - Use `bool` instead of `boolean`.
  - Use `dict` instead of `dictionary`.

- Square brackets should be used for the dictionary type. For example:

```python
my_var (dict[str, int]): Variable description.
```

- If a variable has two different possible types, then the word `or` should be used without a comma. Otherwise variables with 3 or more types should use commas to separate the types. Example:

```python
x (type1 or type2): Variable description.
y (type1, type2, or type3): Variable description.
```

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html), them using the following template, which is similar to the [PyTorch Docstring](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#docstring-type-formatting):

```python
def my_function(arg1: type1, arg2: type2, my_var: int = 1) -> returntype:
    """Short description of the function.

    (Optional) Long description of the function or example usage.

    Args:
        arg1 (type1): Variable description.
        arg2 (type2): Variable description.
        my_var (int, optional): Variable description. Default: 1

    Returns:
        Variable description.
    """
    ...
    return my_variable
```

## Pull Request (PR)

Before proposing a PR, please open an issue, where the feature will be discussed. This prevent from duplicated PR to be proposed and also ease the code review process.

Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/).
