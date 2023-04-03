
import sys
import os
from easydict import EasyDict
package_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, package_path)
from vision_base.utils.utils import cfg_from_file, update_cfg
from vision_base.utils.builder import build

import pytest
def test_cfg():
    cfg_folder = "tests/example_cfgs"
    for file in os.listdir(cfg_folder):
        if file.endswith(".py"):
            cfg = cfg_from_file(os.path.join(cfg_folder, file))
            assert isinstance(cfg, EasyDict)

def test_merge_cfg():
    cfg = {'a':1,
            'b':{'c':0, 'f':2},
            'c':3
            }
    kwargs={
        'a': 2,
        'b.c': 3,
        'd.e.f': 4,
        'c.g': 1
    }
    cfg = update_cfg(cfg, **kwargs)
    # original
    assert(cfg['b']['f'] == 2)
    # direct updates
    assert(cfg['a'] == 2)
    assert(cfg['b']['c'] == 3)
    # Create new sub dictionary
    assert(isinstance(cfg['d']['e'], dict))
    assert(cfg['d']['e']['f'] == 4)
    # Overwrite datatype of existing value
    assert(isinstance(cfg['c'], dict))
    assert(cfg['c']['g'] == 1)