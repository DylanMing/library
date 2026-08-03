# 总览

Git 是一个版本控制系统，适合于中小型项目，也是我们最常见的。
github和gitee则是使用git的代码托管平台。

## 基本流程
当我们使用这些代码托管平台时，一个一般的流程如下：
1. 在平台上**创建一个仓库**，这个仓库就属于remote端共享的仓库
2. 将仓库**克隆到本地**`git clone https://github.com/...`仓库的地址由代码托管平台给出

这样我们就将远端的仓库clone了一份到本地，这样就**有git的配置信息了**
需要注意的是，**本地仓库和本地文件并不是等价的**，即便文件在本地仓库的目录，但是本地仓库的配置信息不会默认将其包含在内，**仍然需要我们将文件提交到本地仓库**，然后**再推送到远端**。


当然，仓库是有**不同分支**的，默认的分支为main（master），我们可以新建不同的分支，这样做的好处就是可以分开不同人提交的代码，对不同的版本进行控制，方便我们进行更新和改进。

一个基本使用：在与远程仓库连接后，把本地修改同步到远端
```bash
git add .
git commit -m "commit message"
git push
```

远程仓库和本地有冲突

```bash
git fetch origin
git merge origin/master
git pull origin master
```



## 基础命令

### 分支

新建分支

```bash
git branche <分支名>
```

切换分支

```bash
git checkout <分支名>
```


新建分支并切换分支可以通过一条命令

```bash
git checkout -b <分支名>
```

合并分支（需要在main操作）

```bash
git merge <分支名>
```

或者rebase,将分支删除并合并（需要在分支内操作），再rebase将其合并到main

```bash
git rebase main
git rebase <分支名>
```

### head移动

head是指向提交记录的指针，默认是指向分支的最新节点，但是我们可以手动切换到之前的提交记录

```bash
git checkou <分支名>^ # 切换到分支最新节点的上一个节点
git checkou <分支名>~n # 切换到分支最新节点的上n个节点
```



`git diff`可以显示当前本地文件和仓库之间有什么区别

`git add <changed_file>` 可以将需要加入到仓库的本地文件加入暂存区

`git commit`则可以把暂存区里的问价提交到本地仓库

`git push origin my-feature` 此时会将my-feature分支更新到代码托管平台（原来平台上没有my-feature这个分支）


而有的时候，main分支在我们提交新的分支前又有更新了，那么此时需要先同步main分支的更新到本地：

`git checkout main` 切换到`main` 分支中，此时再运行`git pull origin master` ，就把远端的main同步到了本地的main分支中，然后再回到my-feature分支`git checkout my-feature`

这时候就需要合并分支了，使用`git rebase main`， 就会在my-feature分支上先添加远端main的修改，再尝试合并之前它和本地的修改。如果出现`rebase conflit` 则说明有冲突，需要手动选择。

最后`git push -f origin my-feature` 将本地仓库的branch push到代码托管平台上，因为有rebase，所以-f强制push


而最后将我们更新的代码合并到远端的main上，这个过程叫`pull request` 
在仓库主确认后，会将合并的代码做`squash and merge` 即将分支的所有改变转变成一个改变后合并。

然后删除my-feature分支，远端一般有按钮，本地使用`git branch -D my-feature`

最后将远端更新拉到本地`git pull origin master`


下面是git的流程图，
	![[Pasted image 20221002185919.png]]

# 新建仓库并初次上传

在github或者gitee上先新建一个仓库，


本地操作：
将文件夹初始化为本地git仓库

```bash
git init
```

此时本地文件夹初始化为git仓库了，也产生了本地仓库，暂存区和工作区
下一步是把本地文件夹中的内容添加到暂存区中

`git add .` 会把所有文件都添加到暂存区中，也可以选择想上传的文件添加到暂存区
```bash
git add .
```

`git commit -m "commit message"` 暂存区提交到本地仓库

```bash
git commit -m "commit message"
```

然后需要将本地仓库连接到远程仓库,远程仓库地址可以在github或者gitee中获得

```bash
git remote add origin https://github.com/[]/[].git
```

将分支重命名为main
```bash
git branch -M main
```

推送到远程分支，origin是远程仓库的名字，可以修改，main是分支的名称
```bash
git push -u origin main
```

查看当前修改文件

```bash
git status
```







# 拉取远端更新到本地

如果是第一次下载，本地没有该仓库的代码，直接 `git clone` 就行

如果本地已有代码，那么拉取远端更新到本地一般有两种路径
- `git fetch` 把远端仓库拉取到本地仓库，再


# 查看git记录
[GitHub（三）本地仓库：git log 查看项目历史的 commit 记录 - 风影忍着的文章 - 知乎](https://zhuanlan.zhihu.com/p/653978516)


git log 查看项目历史的 commit 记录

可以看到commit ID, 作者，日期等信息，head表示目前在哪一个分支上

```bash
git log

commit 7d447ca9999bba505bef6101aa7f562a42f9dfb0 (HEAD -> main)
Author: Dylanming <littlebright666@gmail.com>
Date:   Sun Jun 16 11:32:34 2024 +0800

    change before add edge feature

commit d1a3e0c062e5c5855f268df2b286919502a4b72d (origin/main, origin/HEAD)
Merge: a2286fe e1372ef
Author: Dylanming <littlebright666@gmail.com>
:...skipping...
commit 7d447ca9999bba505bef6101aa7f562a42f9dfb0 (HEAD -> main)
Author: Dylanming <littlebright666@gmail.com>
Date:   Sun Jun 16 11:32:34 2024 +0800

    change before add edge feature

commit d1a3e0c062e5c5855f268df2b286919502a4b72d (origin/main, origin/HEAD)
Merge: a2286fe e1372ef
Author: Dylanming <littlebright666@gmail.com>
Date:   Wed May 22 12:43:48 2024 +0800

    Merge branch 'main' of https://github.com/DylanMing/GENNET

:...skipping...
commit 7d447ca9999bba505bef6101aa7f562a42f9dfb0 (HEAD -> main)
Author: Dylanming <littlebright666@gmail.com>
Date:   Sun Jun 16 11:32:34 2024 +0800

    change before add edge feature

commit d1a3e0c062e5c5855f268df2b286919502a4b72d (origin/main, origin/HEAD)
Merge: a2286fe e1372ef
Author: Dylanming <littlebright666@gmail.com>
Date:   Wed May 22 12:43:48 2024 +0800

    Merge branch 'main' of https://github.com/DylanMing/GENNET

commit a2286fed749967fbb39acb1ae90cc4fdd11c6909
Author: Dylanming <littlebright666@gmail.com>
Date:   Wed May 22 11:29:23 2024 +0800

    5.22 train

commit e1372eff6edbddd48dd80102ac0c9218c0c5846b
Author: DylanMing <littlebright666@gmail.com>

```

简短信息可以使用：

```bash
git log --pretty=oneline
```

# git 本地仓库版本回退

回退到某一次commit版本

```bash
git reset --hard commitID
```



回退后后悔了。可以用下面的查看所有commit和reset记录
```bash
git reflog
```

然后再次reset即可


# git 代理
		
设置代理

```
git config --global http.proxy http://127.0.0.1:10809
git config --global https.proxy http://127.0.0.1:10809
```

取消代理

```
git config --global --unset http.proxy
git config --global --unset https.proxy
```


# git 本地配置

`git config` 工具专门用来配置或读取相应的工作环境变量。而正是由这些[环境变量](https://www.zhihu.com/search?q=%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%22683245248%22%7D)，决定了 Git 在各个环节的具体工作方式和行为。这些变量可以存放在以下三个不同的地方：

1. `/etc/gitconfig` 文件：包含系统上每一个用户及他们仓库的通用配置。若使用 `git config` 时带上 `--system` 选项，读写的就是这个文件。  
2. `~/.gitconfig` 文件：用户目录下的配置文件只适用于该用户。若使用 `git config` 时带上 `--global` 选项，读写的就是这个文件。  
3. 当前仓库的 Git 目录中的配置文件（也就是工作目录中的 `.git/config` 文件）：这里的配置仅仅针对当前仓库有效。若使用 `git config` 时带上 `--local` 选项，读写的就是这个文件。  
4. 
**每一个级别的配置都会覆盖上层的相同配置，所以 `.git/config` 里的配置会覆盖 `/etc/gitconfig` 中的同名变量。**

你可以通过以下命令查看所有的配置以及它们所在的文件： 带上 `--[show-origin]` 可以显示配置所在文件目录


```bash
git config --list --show-origin
```



对于全局账户的配置
```bash
git config --global user.email "mail address"   
git config --global user.name "Name" 
```

对于单独项目的配置

在该目录下使用：
```bash
git config user.name <name>
git config user.email <email>
```

# 快速查看不同

用下面命令可以快速查看本地和远端的不同
```bash
git diff origin/master
```

# git冲突处理
#todo


# 添加忽略文件.gitignore

首先需要再本地更新 `.gitignore` 文件

然后删除本地仓库中的文件，再重新添加后提交

```bash
git rm -r --cached .
git add .
git commit -m "refresh cache and rebuild index for .gitignore valid"
```

由于你正在修改历史记录，你需要使用 `git push` 命令的 `--force` 选项来强制推送到远程仓库。这将覆盖远程仓库的历史记录，因此请谨慎使用，并确保你的团队成员都了解这一更改

```bash
git push 
```

更新团队成员的本地仓库： 通知你的团队成员，他们需要使用以下命令来更新他们的本地仓库：

```bash
git fetch origin
git reset --hard origin/branchname
```

比如要忽略文件目录下所有的 `.DS_Store` 那么就可以添加

```
**/.DS_Store
```


# git 本地分支和远程名称不同
![[Pasted image 20250227182328.png]]

https://blog.csdn.net/owo_ovo/article/details/135176894

将分支重命名即可
```
git branch -m master
```

# git remote 和远程仓库相关

`git remote`命令用来创建、查看和删除本地仓库与其他代码仓库之间的连接

## 查看连接
这个命令会显示当前所有的远程链接，包括远程仓库的名称和对应的URL。
```bash
git remote -v
```

## 切断连接


切断指定连接

```bash
git remote remove [远程仓库名称]
```

切断所有连接

```bash
git remote rm $(git remote)
```


## 添加连接

其实这个命令就是修改 `./.git/config` 文件，

```bash
git remote add <name> <url>
```

上面的命令创建了一个与远端仓库的关联关系。在此之后，你就可以使用`<name>`作为这个仓库的别名在其他git命令中使用。

每当你使用`git clone`命令clone一个远端的仓库，都会自动创建一个remote链接叫做origin，并指回被clone的远端仓库。由于这一操作为获取上游变更或者提交本地变更提供了快捷方式，于是通过此命令在本地创建中心仓库的副本成为开发者的常见操作。这种默认创建origin上游的行为也是很多托管在git上的项目称自己的中心仓库为origin的原因。


## 重命名

```bash
git remote rename <old-name> <new-name>
```



