# Description

Describe your changes in detail.

## Motivation and Context

Why is this change required? What problem does it solve?
If it fixes an open issue, please link to the issue here.
You can use the syntax `close #1314520` if this solves the issue #15213

- [ ] I have raised an issue to propose this change ([required](https://github.com/PKU-Alignment/align-anything/issues) for new features and bug fixes)

## Test

Please test your changes by running the following command:

```bash
cd scripts
bash test/test_text_to_text.sh ./opt PATH_TO_OUTPUT_ROOT_DIR
```

Here, `./opt` is the directory containing the test scripts for the `opt` model, and `PATH_TO_OUTPUT_ROOT_DIR` is the path to the output root directory. The test scripts will save ~1GB data to the output root directory and delete it after the test. Please ensure you have enough space on your disk.

## Lint

Please run the following command in the root directory to check your code style:

```bash
pip install pre-commit
pre-commit run --all-files
```

## Types of changes

What types of changes does your code introduce? Put an `x` in all the boxes that apply:

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds core functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation (update in the documentation)

## Checklist

Go over all the following points, and put an `x` in all the boxes that apply.
If you are unsure about any of these, don't hesitate to ask. We are here to help!

- [ ] I have read the [CONTRIBUTION](https://github.com/PKU-Alignment/align-anything/blob/HEAD/.github/CONTRIBUTING.md) guide. (**required**)
- [ ] My change requires a change to the documentation.
- [ ] I have updated the tests accordingly. (*required for a bug fix or a new feature*)
- [ ] I have updated the documentation accordingly.
