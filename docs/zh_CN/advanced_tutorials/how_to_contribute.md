# PaddleClas 社区贡献指南
------

## 目录

- [1. 如何贡献代码](#1)
    - [1.1 PaddleClas 分支说明](#1.1)
    - [1.2 PaddleClas 代码提交流程与规范](#1.2)
        - [1.2.1 fork 和 clone 代码](#1.2.1)
        - [1.2.2 和远程仓库建立连接](#1.2.2)
        - [1.2.3 创建本地分支](#1.2.3)
        - [1.2.4 使用 pre-commit 勾子](#1.2.4)
        - [1.2.5 修改与提交代码](#1.2.5)
        - [1.2.6 保持本地仓库最新](#1.2.6)
        - [1.2.7 push 到远程仓库](#1.2.7)
        - [1.2.8 提交 Pull Request](#1.2.8)
        - [1.2.9 签署 CLA 协议和通过单元测试](#1.2.9)
        - [1.2.10 删除分支](#1.2.10)
        - [1.2.11 提交代码的一些约定](#1.2.11)
- [2. 总结](#2)
- [3. 参考文献](#3)

<a name="1"></a>
## 1. 如何贡献代码

<a name="1.1"></a>
### 1.1 PaddleClas 分支说明

PaddleClas 未来将维护 2 种分支，分别为：

* release/x.x 系列分支：为稳定的发行版本分支，会适时打 tag 发布版本，适配 Paddle 的 release 版本。当前最新的分支为 release/2.3 分支，是当前默认分支，适配 Paddle v2.1.0 。随着版本迭代， release/x.x 系列分支会越来越多，默认维护最新版本的 release 分支，前 1 个版本分支会修复 bug，其他的分支不再维护。
* develop 分支：为开发分支，适配 Paddle 的 develop 版本，主要用于开发新功能。如果有同学需要进行二次开发，请选择 develop 分支。为了保证 develop 分支能在需要的时候拉出 release/x.x 分支， develop 分支的代码只能使用 Paddle 最新 release 分支中有效的 api 。也就是说，如果 Paddle develop 分支中开发了新的 api，但尚未出现在 release 分支代码中，那么请不要在 PaddleClas 中使用。除此之外，对于不涉及 api 的性能优化、参数调整、策略更新等，都可以正常进行开发。

PaddleClas 的历史分支，未来将不再维护。考虑到一些同学可能仍在使用，这些分支还会继续保留：

* release/static 分支：这个分支曾用于静态图的开发与测试，目前兼容 >=1.7 版本的 Paddle 。如果有特殊需求，要适配旧版本的 Paddle，那还可以使用这个分支，但除了修复 bug 外不再更新代码。
* dygraph-dev 分支：这个分支将不再维护，也不再接受新的代码，请使用的同学尽快迁移到 develop 分支。


PaddleClas 欢迎大家向 repo 中积极贡献代码，下面给出一些贡献代码的基本流程。

<a name="1.2"></a>
### 1.2 PaddleClas 代码提交流程与规范

<a name="1.2.1"></a>
#### 1.2.1 fork 和 clone 代码

* 跳转到 [PaddleClas GitHub 首页](https://github.com/PaddlePaddle/PaddleClas)，然后单击 Fork 按钮，生成自己目录下的仓库，比如 `https://github.com/USERNAME/PaddleClas` 。


![](../../images/quick_start/community/001_fork.png)

* 将远程仓库 clone 到本地

```shell
# 拉取 develop 分支的代码
git clone https://github.com/USERNAME/PaddleClas.git -b develop
cd PaddleClas
```

clone 的地址可以从下面获取

![](../../images/quick_start/community/002_clone.png)

<a name="1.2.2"></a>
#### 1.2.2 和远程仓库建立连接

首先通过 `git remote -v` 查看当前远程仓库的信息。

```
origin    https://github.com/USERNAME/PaddleClas.git (fetch)
origin    https://github.com/USERNAME/PaddleClas.git (push)
```

上面的信息只包含了 clone 的远程仓库的信息，也就是自己用户名下的 PaddleClas，接下来我们创建一个原始 PaddleClas 仓库的远程主机，命名为 upstream 。

```shell
git remote add upstream https://github.com/PaddlePaddle/PaddleClas.git
```

使用 `git remote -v` 查看当前远程仓库的信息，输出如下，发现包括了 origin 和 upstream 2 个远程仓库。

```
origin    https://github.com/USERNAME/PaddleClas.git (fetch)
origin    https://github.com/USERNAME/PaddleClas.git (push)
upstream    https://github.com/PaddlePaddle/PaddleClas.git (fetch)
upstream    https://github.com/PaddlePaddle/PaddleClas.git (push)
```

这主要是为了后续在提交 pull request(PR)时，始终保持本地仓库最新。

<a name="1.2.3"></a>
#### 1.2.3 创建本地分支

可以基于当前分支创建新的本地分支，命令如下。

```shell
git checkout -b new_branch
```

也可以基于远程或者上游的分支创建新的分支，命令如下。

```shell
# 基于用户远程仓库(origin)的 develop 创建 new_branch 分支
git checkout -b new_branch origin/develop
# 基于上游远程仓库(upstream)的 develop 创建 new_branch 分支
# 如果需要从 upstream 创建新的分支，需要首先使用 git fetch upstream 获取上游代码
git checkout -b new_branch upstream/develop
```

最终会显示切换到新的分支，输出信息如下

```
Branch new_branch set up to track remote branch develop from upstream.
Switched to a new branch 'new_branch'
```

<a name="1.2.4"></a>
#### 1.2.4 使用 pre-commit 勾子

Paddle 开发人员使用 pre-commit 工具来管理 Git 预提交钩子。 它可以帮助我们格式化源代码(C++，Python)，在提交(commit)前自动检查一些基本事宜（如每个文件只有一个 EOL，Git 中不要添加大文件等）。

pre-commit 测试是 Travis-CI 中单元测试的一部分，不满足钩子的 PR 不能被提交到 PaddleClas，首先安装并在当前目录运行它：

```shell
pip install pre-commit
pre-commit install
```

* **注意**

1. Paddle 使用 clang-format 来调整 C/C++ 源代码格式，请确保 `clang-format` 版本在 3.8 以上。
2. 通过 `pip install pre-commit` 和 `conda install -c conda-forge pre-commit` 安装的 `yapf` 稍有不同的，PaddleClas 开发人员使用的是 `pip install pre-commit` 。

<a name="1.2.5"></a>
#### 1.2.5 修改与提交代码

可以通过 `git status` 查看改动的文件。
对 PaddleClas 的 `README.md` 做了一些修改，希望提交上去。则可以通过以下步骤

```shell
git add README.md
pre-commit
```

重复上述步骤，直到 pre-comit 格式检查不报错。如下所示。

![](../../images/quick_start/community/003_precommit_pass.png)

使用下面的命令完成提交。

```shell
git commit -m "your commit info"
```

<a name="1.2.6"></a>
#### 1.2.6 保持本地仓库最新

获取 upstream 的最新代码并更新当前分支。这里的 upstream 来自于 1.2 节的`和远程仓库建立连接`部分。

```shell
git fetch upstream
# 如果是希望提交到其他分支，则需要从 upstream 的其他分支 pull 代码，这里是 develop
git pull upstream develop
```

<a name="1.2.7"></a>
#### 1.2.7 push 到远程仓库

```shell
git push origin new_branch
```

<a name="1.2.8"></a>
#### 1.2.8 提交 Pull Request

点击 new pull request，选择本地分支和目标分支，如下图所示。在 PR 的描述说明中，填写该 PR 所完成的功能。接下来等待 review，如果有需要修改的地方，参照上述步骤更新 origin 中的对应分支即可。

![](../../images/quick_start/community/004_create_pr.png)
<a name="1.2.9"></a>
#### 1.2.9 签署 CLA 协议和通过单元测试

* 签署 CLA
在首次向 PaddlePaddle 提交 Pull Request 时，您需要您签署一次 CLA(Contributor License Agreement)协议，以保证您的代码可以被合入，具体签署方式如下：

1. 请您查看 PR 中的 Check 部分，找到 license/cla，并点击右侧 detail，进入 CLA 网站
2. 点击 CLA 网站中的 `Sign in with GitHub to agree`, 点击完成后将会跳转回您的 Pull Request 页面

<a name="1.2.10"></a>
#### 1.2.10 删除分支

* 删除远程分支

在 PR 被 merge 进主仓库后，我们可以在 PR 的页面删除远程仓库的分支。

也可以使用 `git push origin :分支名` 删除远程分支，如：


```shell
git push origin :new_branch
```

* 删除本地分支

```shell
# 切换到 develop 分支，否则无法删除当前分支
git checkout develop

# 删除 new_branch 分支
git branch -D new_branch
```

<a name="1.2.11"></a>
#### 1.2.11 提交代码的一些约定

为了使官方维护人员在评审代码时更好地专注于代码本身，请您每次提交代码时，遵守以下约定：

1）请保证 Travis-CI 中单元测试能顺利通过。如果没过，说明提交的代码存在问题，官方维护人员一般不做评审。

2）提交 Pull Request 前：

请注意 commit 的数量。

原因：如果仅仅修改一个文件但提交了十几个 commit，每个 commit 只做了少量的修改，这会给评审人带来很大困扰。评审人需要逐一查看每个 commit 才能知道做了哪些修改，且不排除 commit 之间的修改存在相互覆盖的情况。

建议：每次提交时，保持尽量少的 commit，可以通过 `git commit --amend` 补充上次的 commit 。对已经 Push 到远程仓库的多个 commit，可以参考 [squash commits after push](https://stackoverflow.com/questions/5667884/how-to-squash-commits-in-git-after-they-have-been-pushed)。

请注意每个 commit 的名称：应能反映当前 commit 的内容，不能太随意。

3）如果解决了某个 Issue 的问题，请在该 Pull Request 的第一个评论框中加上： `fix #issue_number`，这样当该 Pull Request 被合并后，会自动关闭对应的 Issue 。关键词包括： close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved，请选择合适的词汇。详细可参考 [Closing issues via commit messages](https://help.github.com/articles/closing-issues-via-commit-messages)。

此外，在回复评审人意见时，请您遵守以下约定：

1）官方维护人员的每一个 review 意见都希望得到回复，这样会更好地提升开源社区的贡献。

- 对评审意见同意且按其修改完的，给个简单的 Done 即可；
- 对评审意见不同意的，请给出您自己的反驳理由。

2）如果评审意见比较多,

- 请给出总体的修改情况。
- 请采用 `start a review` 进行回复，而非直接回复的方式。原因是每个回复都会发送一封邮件，会造成邮件灾难。

<a name="2"></a>
## 2. 总结

* 开源社区依赖于众多开发者与用户的贡献和反馈，在这里感谢与期待大家向 PaddleClas 提出宝贵的意见与 Pull Request，希望我们可以一起打造一个领先实用全面的图像识别代码仓库！

<a name="3"></a>
## 3. 参考文献
1. [PaddlePaddle 本地开发指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/08_contribution/index_cn.html)
2. [向开源框架提交 pr 的过程](https://blog.csdn.net/vim_wj/article/details/78300239)
