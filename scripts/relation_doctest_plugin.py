"""Pytest items for xdoctest examples, preserving actual source file and example ID."""
import os
from pathlib import Path
import pytest


class Example(pytest.Item):
    def __init__(self,*args,example,**kwargs):
        super().__init__(*args,**kwargs);self.example=example
    def runtest(self):
        self.example.run(verbose=0,on_error='raise')
    def reportinfo(self):
        return self.path,max(self.example.lineno-1,0),self.name


class Examples(pytest.File):
    def collect(self):
        from xdoctest import core
        for example in core.parse_doctestables(str(self.path),style='google',analysis='static'):
            example.config.update(global_exec='from torch import nn\nimport torch.nn.functional as F\nimport torch\nimport warnings',
                                  default_runtime_state={'IGNORE_WHITESPACE':True})
            yield Example.from_parent(self,name=example.unique_callname,example=example)


def pytest_collect_file(file_path,parent):
    if file_path.suffix=='.py' and not any(part in ('distributed','_vendor') for part in file_path.parts):
        return Examples.from_parent(parent,path=file_path)
