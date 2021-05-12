# Context to load scripts from other directory
import os
from contextlib import contextmanager

@contextmanager
def dir_change(path):
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)
