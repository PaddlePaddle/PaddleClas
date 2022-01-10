# How to Contribute to the PaddleClas Community

------

## Catalogue

- [1. How to Contribute Code](#1)
  - [1.1 Branches of PaddleClas](#1.1)
  - [1.2 Commit Code to PaddleClas](#1.2)
    - [1.2.1 Codes of Fork and Clone](#1.2.1)
    - [1.2.2 Connect to the Remote Repository](#1.2.2)
    - [1.2.3 Create the Local Branch](#1.2.3)
    - [1.2.4 Employ Pre-commit Hook](#1.2.4)
    - [1.2.5 Modify and Commit Code](#1.2.5)
    - [1.2.6 Keep the Local Repository Updated](#1.2.6)
    - [1.2.7 Push to Remote Repository](#1.2.7)
    - [1.2.8 Commit Pull Request](#1.2.8)
    - [1.2.9 CLA and Unit Test](#1.2.9)
    - [1.2.10 Delete Branch](#1.2.10)
    - [1.2.11 Conventions](#1.2.11)
- [2. Summary](#2)
- [3. Inferences](#3)


<a name="1"></a>
## 1. How to Contribute Code


<a name="1.1"></a>
### 1.1 Branches of PaddleClas

PaddleClas maintains the following two branches:

- release/x.x series: Stable release branches, which are tagged with the release version of Paddle in due course.
The latest and the default branch is the release/2.3, which is compatible with Paddle v2.1.0.
The branch of release/x.x series will continue to grow with future iteration,
and the latest release will be maintained by default, while the former one will fix bugs with no other branches covered.
- develop : developing branch, which is adapted to the develop version of Paddle and is mainly used for
developing new functions. A good choice for secondary development.
To ensure that the develop branch can pull out the release/x.x when needed,
only the API that is valid in Paddle's latest release branch can be adopted for its code.
In other words, if a new API has been developed in this branch but not yet in the release,
please do not use it in PaddleClas. Apart from that, features that do not involve the performance optimizations,
parameter adjustments, and policy updates of the API can be developed normally.

The historical branches of PaddleClas will not be maintained, but will be remained for the existing users.

- release/static: This branch was used for static graph development and testing,
and is currently compatible with >=1.7 versions of Paddle.
It is still practicable for the special need of adapting an old version of Paddle,
but the code will not be updated except for bug fixing.
- dygraph-dev: This branch will no longer be maintained and accept no new code.
Please transfer to the develop branch as soon as possible.

PaddleClas welcomes code contributions to the repo, and the basic process is detailed in the next part.


<a name="1.2"></a>
### 1.2 Commit the Code to PaddleClas


<a name="1.2.1"></a>
#### 1.2.1 Codes of Fork and Clone

- Skip to the home page of [PaddleClas GitHub](https://github.com/PaddlePaddle/PaddleClas) and click the
Fork button to generate a repository in your own directory, such as `https://github.com/USERNAME/PaddleClas`.

[](../../images/quick_start/community/001_fork.png)

- Clone the remote repository to local

```shell
# Pull the code of the develop branch
git clone https://github.com/USERNAME/PaddleClas.git -b develop
cd PaddleClas
```

Obtain the address below

[](../../images/quick_start/community/002_clone.png)


<a name="1.2.2"></a>
#### 1.2.2 Connect to the Remote Repository

First check the current information of the remote repository with `git remote -v`.

```shell
origin    https://github.com/USERNAME/PaddleClas.git (fetch)
origin    https://github.com/USERNAME/PaddleClas.git (push)
```

The above information only contains the cloned remote repository,
which is the PaddleClas under your username. Then we create a remote host of the original PaddleClas repository named upstream.

```shell
git remote add upstream https://github.com/PaddlePaddle/PaddleClas.git
```

Adopt `git remote -v` to view the current information of the remote repository,
and 2 remote repositories including origin and upstream can be found, as shown below.

```shell
origin    https://github.com/USERNAME/PaddleClas.git (fetch)
origin    https://github.com/USERNAME/PaddleClas.git (push)
upstream    https://github.com/PaddlePaddle/PaddleClas.git (fetch)
upstream    https://github.com/PaddlePaddle/PaddleClas.git (push)
```

This is mainly to keep the local repository updated when committing a pull request (PR).


<a name="1.2.3"></a>
#### 1.2.3 Create the Local Branch

Run the following command to create a new local branch based on the current one.

```shell
git checkout -b new_branch
```

Or you can create new ones based on remote or upstream branches.

```shell
# Create the new_branch based on the develope of origin (unser remote repository)
git checkout -b new_branch origin/develop
# Create the new_branch base on the develope of upstream
# If you need to create a new branch from upstream,
# please first employ git fetch upstream to fetch the upstream code
git checkout -b new_branch upstream/develop
```

The following output shows that it has switched to the new branch with :

```
Branch new_branch set up to track remote branch develop from upstream.
Switched to a new branch 'new_branch'
```


<a name="1.2.4"></a>
#### 1.2.4 Employ Pre-commit Hook

Paddle developers adopt the pre-commit tool to manage Git pre-commit hooks.
It helps us format the source code (C++, Python) and automatically check basic issues before committing
e.g., one EOL per file, no large files added to Git, etc.

The pre-commit test is part of the unit tests in Travis-CI,
and PRs that do not satisfy the hook cannot be committed to PaddleClas.
Please install it first and run it in the current directory:

```
pip install pre-commit
pre-commit install
```

- **Note**

1. Paddle uses clang-format to format C/C++ source code, please make sure `clang-format` has a version of 3.8 or higher.
2. `yapf` installed by `pip install pre-commit` and `conda install -c conda-forge pre-commit` is slightly different,
and the former one is chosen by PaddleClas developers.


<a name="1.2.5"></a>
#### 1.2.5 Modify and Commit Code

You can check the changed files via `git status`. Follow the steps below to commit the `README.md` of PaddleClas after modification:

```
git add README.md
pre-commit
```

Repeat the above steps until the pre-commit format check does not report an error, as shown below.

[](../../images/quick_start/community/003_precommit_pass.png)

Run the following command to commit.

```
git commit -m "your commit info"
```


<a name="1.2.6"></a>
#### 1.2.6 Keep the Local Repository Updated

Get the latest code for upstream and update the current branch.
The upstream here is from the `Connecting to a remote repository` part in section 1.2.

```
git fetch upstream
# If you want to commit to another branch, please pull the code from another branch of upstream, in this case it is develop
git pull upstream develop
```


<a name="1.2.7"></a>
#### 1.2.7 Push to Remote Repository

```
git push origin new_branch
```


<a name="1.2.8"></a>
#### 1.2.8 Commit Pull Request

Click new pull request and select the local branch and the target branch,
as shown in the following figure. In the description of the PR, fill out what the PR accomplishes.
Next, wait for the review, and if any changes are required,
update the corresponding branch in origin by referring to the above steps.

[](../../images/quick_start/community/004_create_pr.png)


<a name="1.2.9"></a>
#### 1.2.9 CLA and Unit Test

- When you first commit a Pull Request to PaddlePaddle,
you are required to sign a CLA (Contributor License Agreement) to ensure that your code can be merged,
please follow the step below to sign CLA:

1. Please examine the Check section of your PR, find license/cla,
and click the detail on the right side to enter the CLA website
2. Click `Sign in with GitHub to agree` on the CLA website,
and you will be redirected back to your Pull Request page when you are done.


<a name="1.2.10"></a>
#### 1.2.10 Delete Branch

- Delete remote branch

When the PR is merged into the main repository, you can delete the remote branch from the PR page.

You can also delete the remote branch using `git push origin :branch name`, e.g.

```
git push origin :new_branch
```

- Delete local branch

```
# Switch to the develop branch, otherwise the current branch cannot be deleted
git checkout develop

# Delete new_branch
git branch -D new_branch
```


<a name="1.2.11"></a>
#### 1.2.11 Conventions

To help official maintainers focus on the code itself when reviewing it,
please adhere to the following conventions each time you commit code:

1. Please pass the unit test in Travis-CI first.
Otherwise, the submitted code may have problems and usually receive no official review.
2. Before committing a Pull Request:
Note the number of commits.

Reason: If only one file is modified but more than a dozen commits are committed with a few changes for each,
this may overwhelm the reviewer for they need to check each and every commit for specific changes,
including the case that the changes between commits overwrite each other.

Recommendation: Minimize the number of commits each time, and add the last commit with `git commit --amend`.
For multiple commits that have been pushed to a remote repository, please refer to
[squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed).

Please pay attention to the name of each commit:
it should reflect the content of the current commit without being too casual.

3. If an issue is resolved, please add `fix #issue_number` to the first comment box of the Pull Request,
so that the corresponding issue will be closed automatically when the Pull Request is merged. Please choose the appropriate term with keywords such as close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved, please choose the appropriate term. See details in [Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages).

In addition, please stick to the following convention to respond to reviewers' comments:

1. Every review comment from the official maintainer is expected to be answered,
which will better enhance the contribution of the open source community.

-  If you agree with the review and finish the corresponding modification, please simply return Done;
-  If you disagree with the review, please give your reasons.

2. If there are plenty of review comments,

- Please present the revision in general.
- Please reply with `start a review` instead of a direct approach, for it may be overwhelming to receive the email of every reply.


<a name="2"></a>
## 2. Summary

- The open source community relies on the contributions and feedback of developers and users.
We highly appreciate that and look forward to your valuable comments and Pull Requests to PaddleClas in the hope that together we can build a leading practical and comprehensive code repository for image recognition!


<a name="3"></a>
## 3. References

1. [Guide to PaddlePaddle Local Development](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/08_contribution/index_en.html)
2. [Committing PR to Open Source Framework](https://blog.csdn.net/vim_wj/article/details/78300239)
