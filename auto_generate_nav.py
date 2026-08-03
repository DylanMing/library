import os
import re

def natural_sort_key(s):
    """
    返回一个可用于自然语言排序的键。
    将字符串按数字和非数字部分分割，然后转换数字部分为整数。
    """
    # 使用正则表达式来分割字符串，数字部分会被捕获并作为单独的元素
    # 例如: "10.两数之和.md" -> ['', '10', '.', '两数之和', '.md']
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def generate_mkdocs_nav_string(directory='docs', indent_spaces=2):
    """
    遍历指定目录，生成用于MkDocs的导航YAML字符串。
    使用自定义的自然语言排序。

    :param directory: 要扫描的根目录，默认为'docs'。
    :param indent_spaces: 缩进的空格数，默认为2。
    :return: 包含导航结构的YAML字符串。
    """
    nav_lines = ["nav:"]

    def walk_directory(path, level=1):
        indent = ' ' * (level * indent_spaces)
        
        # 获取原始文件列表
        items = os.listdir(path)
        
        # 使用自定义的 natural_sort_key 进行排序
        items.sort(key=lambda x: (not os.path.isdir(os.path.join(path, x)), natural_sort_key(x)))

        for item in items:
            full_path = os.path.join(path, item)

            if item.startswith('.'):
                continue

            if os.path.isfile(full_path) and item.endswith('.md'):
                filename, _ = os.path.splitext(item)
                relative_path = os.path.relpath(full_path, directory).replace('\\', '/')

                # 移除文件名开头的数字和特殊字符，以便生成更干净的标题
                title = re.sub(r'^\d+\s*[\._-]?\s*', '', filename)
                title = title.replace('-', ' ').title()
                if not title:
                     title = filename
                
                nav_lines.append(f"{indent}- '{title}': '{relative_path}'")
            
            elif os.path.isdir(full_path):
                # 移除目录名开头的数字和特殊字符
                title = re.sub(r'^\d+\s*[\._-]?\s*', '', item)
                title = title.replace('-', ' ').title()
                if not title:
                    title = item

                # 检查目录中是否有 index.md 或 README.md
                has_index = 'index.md' in os.listdir(full_path) or 'README.md' in os.listdir(full_path)
                
                nav_lines.append(f"{indent}- '{title}':")
                walk_directory(full_path, level + 1)
                
    walk_directory(directory)
    return '\n'.join(nav_lines)

if __name__ == "__main__":
    docs_dir = 'mkdocs/docs'
    if not os.path.isdir(docs_dir):
        print(f"错误：指定的目录 '{docs_dir}' 不存在。")
    else:
        nav_yaml_string = generate_mkdocs_nav_string(docs_dir)
        print("--- 生成的导航配置如下 ---")
        print(nav_yaml_string)
        with open('generated_nav.yaml', 'w', encoding='utf-8') as f:
            f.write(nav_yaml_string)
        print("配置已保存到 generated_nav.yaml")