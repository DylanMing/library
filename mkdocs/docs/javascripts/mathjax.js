// MathJax 配置：配合 pymdownx.arithmatex (generic: true) 使用
// arithmatex 在 generic 模式下输出 \(...\) 和 \[...\] 分隔符
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processRef: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
