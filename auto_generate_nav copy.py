import os
import re

def natural_sort_key(s):
    """
    Returns a key for natural language sorting.
    Splits the string into numeric and non-numeric parts, then converts
    the numeric parts to integers.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def generate_mkdocs_nav_string(directory='docs', indent_spaces=2):
    """
    Traverses the specified directory to generate the MkDocs nav YAML string.
    
    :param directory: The root directory to scan, defaults to 'docs'.
    :param indent_spaces: The number of spaces for indentation, defaults to 2.
    :return: The generated nav YAML string.
    """
    nav_lines = ["nav:"]

    def walk_directory(path, level=1, is_nested_child=False):
        indent = ' ' * (level * indent_spaces)
        
        items = os.listdir(path)
        items.sort(key=lambda x: (not os.path.isdir(os.path.join(path, x)), natural_sort_key(x)))

        for item in items:
            full_path = os.path.join(path, item)

            if item.startswith('.'):
                continue

            if os.path.isfile(full_path) and item.endswith('.md'):
                filename, _ = os.path.splitext(item)
                relative_path = os.path.relpath(full_path, directory).replace('\\', '/')
                
                # Special handling for top-level index.md and nested index.md
                if filename.lower() in ['index', 'readme']:
                    if level == 1:
                        # Top-level index.md should have an empty title
                        nav_lines.append(f"{indent}- '': '{relative_path}'")
                    else:
                        # Nested index.md should also have an empty title
                        nav_lines.append(f"{indent}- '': '{relative_path}'")
                else:
                    # Regular markdown file
                    title = re.sub(r'^\d+\s*[\._-]?\s*', '', filename)
                    title = title.replace('-', ' ').title()
                    if not title:
                         title = filename
                    nav_lines.append(f"{indent}- '{title}': '{relative_path}'")

            elif os.path.isdir(full_path):
                # Handle nested directories like 'leetcode/leetcode'
                if level == 1 and os.path.basename(full_path) == os.path.basename(os.path.dirname(full_path)):
                    walk_directory(full_path, level, True) # Skip this level and go deeper
                    continue

                title = re.sub(r'^\d+\s*[\._-]?\s*', '', item)
                title = title.replace('-', ' ').title()
                if not title:
                    title = item
                
                nav_lines.append(f"{indent}- '{title}':")
                walk_directory(full_path, level + 1)
                
    walk_directory(directory)
    return '\n'.join(nav_lines)

if __name__ == "__main__":
    # Assuming your docs directory is where the script is run
    docs_dir = 'mkdocs/docs'
    
    # Create dummy directory structure for testing purposes
    # os.makedirs(os.path.join(docs_dir, 'leetcode', 'leetcode'), exist_ok=True)
    # with open(os.path.join(docs_dir, 'index.md'), 'w') as f: f.write('')
    # with open(os.path.join(docs_dir, 'leetcode', 'leetcode', '1.两数之和.md'), 'w') as f: f.write('')
    # with open(os.path.join(docs_dir, 'leetcode', '2.两数相加.md'), 'w') as f: f.write('')
    
    if not os.path.isdir(docs_dir):
        print(f"Error: The directory '{docs_dir}' does not exist. Please create it or change the 'docs_dir' variable.")
    else:
        nav_yaml_string = generate_mkdocs_nav_string(docs_dir)
        print("--- Your generated nav configuration below ---")
        print(nav_yaml_string)
        with open('generated_nav.yaml', 'w', encoding='utf-8') as f:
            f.write(nav_yaml_string)
        print("Configuration has been saved to generated_nav.yaml")