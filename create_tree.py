import os

def generate_tree(startpath, exclude_dirs):
    tree = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f'{subindent}{f}')
    return '\n'.join(tree)

exclude = {'venv311', '.git', '__pycache__', '.pytest_cache', 'node_modules', '.streamlit'}
tree_str = generate_tree(r'c:\Users\km080\OneDrive\Desktop\1', exclude)
with open(r'C:\Users\km080\.gemini\antigravity-ide\brain\c5711e8d-e8d0-450f-8a18-a1996f46724c\project_tree.md', 'w', encoding='utf-8') as f:
    f.write('```\n' + tree_str + '\n```')
print("Tree generated.")
