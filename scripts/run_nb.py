import sys
import nbformat
from nbclient import NotebookClient

def execute_nb(filepath):
    print(f"Executing {filepath} top-to-bottom...")
    nb = nbformat.read(filepath, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': 'notebooks/'}})
    client.execute()
    nbformat.write(nb, filepath)
    print(f"Completed execution of {filepath} successfully!")

if __name__ == '__main__':
    execute_nb(sys.argv[1])
