#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import pytest
import numpy as np
import torch
from decoder import CTCDecoder, GreedyDecoder, CharacterTokenizer

@pytest.fixture(scope='module')
def data():
    mat = torch.Tensor(np.load('data/matrix_probs.npy'))
    classes = CharacterTokenizer.vocab
    data = {'probs': mat, 'classes': classes }
    return(data)

def test_tokenizer(data):
    tok = CharacterTokenizer()
    assert tok.vocab == ["'", '<SPACE>', '<PAD>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    assert tok.text2int("hello") == [9, 6, 13, 13, 16]
    assert tok.int2text([9, 6, 13, 13, 16]) == "hello"
    assert len(tok) == 29

def test_ctc_decoder(data):
    tok = CharacterTokenizer()
    d = CTCDecoder(tok)
    s = "hello"
    seq = tok.text2int('helloooo')
    assert (d.charseq_decode(seq)) == ['h', 'e', 'l', 'o']

def test_greedy(data):
    probs = data['probs']
    classes = data['classes']
    tok = CharacterTokenizer()
    decoder = GreedyDecoder(tok, blank_index=len(tok))
    print(decoder.decode(probs))