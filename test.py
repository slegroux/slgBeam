#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import pytest
import numpy as np
import torch
from IPython import embed
from decoder import CTCDecoder, GreedyDecoder, CharacterTokenizer
from graph import Graph, Node
import copy

@pytest.fixture(scope='module')
def data():
    mat = torch.Tensor(np.load('data/matrix_probs.npy')).t()
    classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # classes = CharacterTokenizer.vocab
    data = {'probs': mat, 'classes': classes }
    return(data)

def test_data(data):
    assert data['probs'].shape ==  torch.Size([80, 100])
    assert len(data['classes']) ==  79

def test_tokenizer(data):
    tok = CharacterTokenizer()
    assert tok.vocab == ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<SPACE>', '<PAD>', '<BOS>', '<EOS>']
    assert tok.text2int("hello") ==  [7, 4, 11, 11, 14]
    assert tok.int2text([7, 4, 11, 11, 14]) == "hello"
    assert len(tok) == 30
    tok = CharacterTokenizer(classes=data['classes'])
    assert tok.vocab == [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    assert tok.text2int("hello") == [60, 57, 64, 64, 67]
    assert tok.int2text([60, 57, 64, 64, 67]) == "hello"
    assert len(tok) == 79

def test_ctc_decoder(data):
    tok = CharacterTokenizer()
    d = CTCDecoder(tok)
    s = "hello"
    seq = tok.text2int('helloooo')
    assert (d.charseq_decode(seq)) == 'helo'

def test_greedy(data):
    tok = CharacterTokenizer(classes=data['classes'])
    decoder = GreedyDecoder(tok, blank_index=len(tok))
    assert decoder.decode(data['probs']) == "the fak friend of the fomly hae tC"

def test_rand(data):
    n1 = Node(1)
    n2 = Node(2)
    nodes = [n1, n2]
    n3 = copy.deepcopy(Node(1))
    nodes.append(n3)
    if n3 in nodes: print("y")
