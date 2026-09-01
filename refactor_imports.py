import os
import re

def update_imports(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements
    new_content = content
    new_content = re.sub(r'from evaluation(?=\.|\s)', r'from src.evaluation', new_content)
    new_content = re.sub(r'import evaluation(?=\.|\s)', r'import src.evaluation', new_content)
    
    new_content = re.sub(r'from restoration(?=\.|\s)', r'from src.restoration', new_content)
    new_content = re.sub(r'import restoration(?=\.|\s)', r'import src.restoration', new_content)
    
    new_content = re.sub(r'from training(?=\.|\s)', r'from src.training', new_content)
    new_content = re.sub(r'import training(?=\.|\s)', r'import src.training', new_content)
    
    new_content = re.sub(r'from utils(?=\.|\s)', r'from src.utils', new_content)
    new_content = re.sub(r'import utils(?=\.|\s)', r'import src.utils as utils', new_content)

    new_content = new_content.replace('from src.utils.preprocessing', 'from src.utils.preprocessing')

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

def main():
    root_dir = '.'
    for dirpath, dnames, fnames in os.walk(root_dir):
        if '.venv' in dirpath or '__pycache__' in dirpath or '.git' in dirpath:
            continue
        for f in fnames:
            if f.endswith('.py'):
                update_imports(os.path.join(dirpath, f))

if __name__ == '__main__':
    main()
