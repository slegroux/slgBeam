#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import pytest
import numpy as np
import torch
from IPython import embed
from beam_search import Tokenizer, Score, BeamSearch

@pytest.fixture(scope='module')
def data():
    mat = torch.Tensor(np.genfromtxt('data/rnnOutput.csv',delimiter=';')[:,: -1])
    mat = mat.unsqueeze(0)
    classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    mat_prob = np.array([[0.2, 0.0, 0.8],
                        [0.4, 0.0, 0.6]])
    syms = 'ab-'
    bs = BeamSearch(syms, mat_prob)
    data = {'probs': mat_prob, 'syms': syms, 'bs': bs }
    return(data)

def test_data(data):
    assert data['probs'].shape ==  (2,3)

def test_tokenizer(data):
    tok = Tokenizer(data['syms'])
    assert(tok.char2int('b') == 1)
    assert(tok.int2char(1) == 'b')

def test_score(data):
    tok = Tokenizer(data['syms'])
    score = Score(tok, data['probs'])
    assert score(1,'-') == 0.6

def test_init(data):
    beam_search = data['bs']
    b, nb, s_b, s_nb = beam_search.init_paths()
    assert b == {''}
    assert nb == {'a', 'b'}
    assert s_b == {'-': 0.8} 
    assert s_nb == {'b': 0.0, 'a': 0.2}

def test_prune(data):
    bs = data['bs']
    path_b, path_nb = bs.prune_paths({''}, {'a','b'}, {'-':0.2}, {'a': 0.1,'b': 0.3}, 2)
    assert path_b == {''}
    assert path_nb == {'b'}
    print(bs.score_b, bs.score_nb)

def test_extend_blank(data):
    bs = data['bs']
    init_b, init_nb, init_s_b, init_s_nb = bs.init_paths()
    print("init:", init_b, init_nb, init_s_b, init_s_nb)
    # incidentally init global b & nb paths
    path_b, path_nb = bs.prune_paths(init_b, init_nb,init_s_b, init_s_nb, 2)
    print("Pruned: ", path_b, path_nb)
    print(bs.score_b, bs.score_nb)
    new_path_b, new_score_b = bs.extend_with_blank(path_b, path_nb, 1)
    print(new_path_b, new_score_b)

def test_extend_syms(data):
    bs = data['bs']
    init_b, init_nb, init_s_b, init_s_nb = bs.init_paths()
    print("init:", init_b, init_nb, init_s_b, init_s_nb)
    # incidentally init global b & nb paths
    path_b, path_nb = bs.prune_paths(init_b, init_nb,init_s_b, init_s_nb, 2)
    print("Pruned: ", path_b, path_nb)
    print(bs.score_b, bs.score_nb)
    new_path_nb, new_score_nb = bs.extend_with_symbol(path_b, path_nb, 1)
    print(new_path_nb, new_score_nb)

def test_merge(data):
    bs = data['bs']
    init_b, init_nb, init_s_b, init_s_nb = bs.init_paths()
    path_b, path_nb = bs.prune_paths(init_b, init_nb,init_s_b, init_s_nb, 2)
    new_path_b, new_score_b = bs.extend_with_blank(path_b, path_nb, 1)
    new_path_nb, new_score_nb = bs.extend_with_symbol(path_b, path_nb, 1)
    bs.merge_paths(new_path_b, new_path_nb)

def test_decode(data):
    bs = data['bs']
    print("decoded: ", bs.decode(2))