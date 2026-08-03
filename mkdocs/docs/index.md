# 销明的个人网站

这里是销明自己尝试使用mkdocs和githubpages构建的个人网站

## 笔记更新工作流程

* 在doc目录下添加新的文件夹或者文件夹内修改markdown文档
* 修改 `mkdocs.yml` 中  `nav` 部分的文件目录

根据markdown构建静态页面，需要进入到 `mkdocs.yml`所在目录

```bash
% cd mkdocs
mkdocs build
```

也可以通过下面的命令在本地先预览一下网站效果

```bash
mkdocs serve
```

将静态页面推送到GitHub部署分支

```bash
mkdocs gh-deploy --force
```
