import pytest
import sys
import os
#sys.path.append(os.path.abspath('../utils'))
from utils.calcoli import somma, conta_unici  

# Tests for somma
def test_somma_integers():
    assert somma(2, 3) == 5
    assert somma(-1, 1) == 0
    assert somma(-5, -7) == -12

def test_somma_floats():
    assert somma(2.5, 3.5) == 6.0
    assert somma(-1.1, 1.1) == 0.0

def test_somma_mixed():
    assert somma(2, 3.5) == 5.5
    assert somma(-2, 3.5) == 1.5

# Tests for conta_unici
def test_conta_unici_normal():
    assert conta_unici([1, 2, 2, 3, 3, 3]) == 3
    assert conta_unici(['a', 'b', 'a', 'c']) == 3
    assert conta_unici([]) == 0

def test_conta_unici_single_element():
    assert conta_unici([1]) == 1
    assert conta_unici(['x']) == 1

def test_conta_unici_all_unique():
    assert conta_unici([1, 2, 3, 4]) == 4
    assert conta_unici(['a', 'b', 'c', 'd']) == 4

def test_conta_unici_mixed_types():
    assert conta_unici([1, '1', 1.0]) == 2
