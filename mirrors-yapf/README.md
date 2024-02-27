yapf mirror
=============

Mirror of yapf package for pre-commit.

For pre-commit: see https://github.com/pre-commit/pre-commit

For yapf: see https://github.com/google/yapf


### Using yapf with pre-commit

Add this to your `.pre-commit-config.yaml`:

    -   repo: git://github.com/pre-commit/mirrors-yapf
        sha: ''  # Use the sha / tag you want to point at
        hooks:
        -   id: yapf
