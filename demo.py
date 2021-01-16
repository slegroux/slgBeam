#!/usr/bin/env python

import BestPath, BeamSearch, LanguageModel
import numpy as np
import pdb

mat = np.load('data/matrix_probs.npy')
print("prob mat shape: ", mat.shape)
classes = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
print("n_classes: ", len(classes))

lm = LanguageModel.LanguageModel('data/line/corpus.txt', classes)
print("bigrams: ", lm.getCharBigram('f','a'))
print("words: ", lm.getWordList())

print("ground truth: ", 'the fake friend of the family, like the')
print('greedy:', '"' + BestPath.ctcBestPath(mat, classes) + '"')
print('beam:', '"' + BeamSearch.ctcBeamSearch(mat, classes, None, beamWidth=25) + '"')
print('beam + lm:', '"' + BeamSearch.ctcBeamSearch(mat, classes, lm) + '"')

