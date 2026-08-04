# library

销明的个人知识库，使用 [Zensical](https://zensical.org/) 构建（Material for MkDocs 团队出品的新一代静态站点生成器），通过 GitHub Actions 自动部署到 GitHub Pages。

线上地址：<https://dylanming.github.io/library/>

## 笔记更新工作流程

1. 在 `mkdocs/docs/` 下新增或修改 markdown 文档
2. 如果调整了目录结构，同步更新 `mkdocs/mkdocs.yml` 中的 `nav` 导航树
3. 推送到 `main` 分支即可，GitHub Actions 会自动构建并部署：

```bash
git add -A
git commit -m "update notes"
git push origin main
```

推送后由 `.github/workflows/deploy.yml` 自动完成构建与发布，无需手动 build。

## 本地预览（可选）

改完想先在本地看效果，用 zensical 的实时预览服务器。

首次使用需配置环境（Python 虚拟环境 + zensical）：

```bash
cd mkdocs
python3 -m venv .venv
.venv/bin/pip install zensical
```

启动预览（修改 markdown 会自动刷新浏览器）：

```bash
.venv/bin/zensical serve -f mkdocs.yml
```

浏览器打开 <http://127.0.0.1:8000/library/>（zensical 按 site_url 在子路径预览，与线上一致；根路径会自动重定向到此），按 `Ctrl+C` 退出。

仅检查能否构建成功（不启动服务）：

```bash
.venv/bin/zensical build -f mkdocs.yml --strict
```

## 目录结构

```
mkdocs/
├── mkdocs.yml          # 站点配置：主题、导航(nav)、插件、扩展（zensical 直接读取）
├── docs/               # 所有 markdown 笔记
│   ├── CSbase/         # 计算机基础
│   ├── leetcode/       # LeetCode 题解
│   ├── 关系抽取/        # NLP 论文笔记
│   ├── 基础知识/
│   └── 杂项/
└── .venv/              # 本地虚拟环境（已 gitignore，不入库）
.github/workflows/
└── deploy.yml          # GitHub Actions 自动部署配置
```

## License

[MIT](LICENSE)
