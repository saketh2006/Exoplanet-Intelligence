import os
import sys
import nbformat
from nbclient import NotebookClient

notebooks = [
    os.path.join('notebooks', '01_preprocessing.ipynb'),
    os.path.join('notebooks', '02_regression.ipynb'),
    os.path.join('notebooks', '03_classification.ipynb')
]

for nb_path in notebooks:
    print(f"\n==========================================")
    print(f"Executing: {nb_path} ...")
    print(f"==========================================")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved: {nb_path} with all outputs embedded!")

print("\nAll 3 notebooks executed top-to-bottom successfully!")
