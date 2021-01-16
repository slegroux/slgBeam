#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

import string
import numpy as np
# score non-blank paths
from IPython import embed


class Tokenizer():
    def __init__(self, symbols):
        # symbol_set = list(string.ascii_lowercase) + ['<bos>','<eos>', '<blank>']
        self._int2char = symbols
        self._char2int = {}
        for i,v in enumerate(self._int2char):
            self._char2int[v] = i
    
    def char2int(self, char):
        return(self._char2int[char])
    
    def int2char(self,int):
        return(self._int2char[int])
    
    def __repr__(self) -> str:
        return(f'{self._char2int}\n{self._int2char}')


class Score():
    def __init__(self, tokenizer, mat_prob):
        self.tokenizer = tokenizer
        self.probs = mat_prob

    def __call__(self, t, c):
        return(self.probs[t][self.tokenizer.char2int(c)])


class BeamSearch():
    def __init__(self, symbols, mat_prob):
        self.score_nb = {}
        self.score_b = {}
        self.symbols = symbols
        self.blank = symbols[-1]
        tok = Tokenizer(self.symbols)
        self.mat_prob = mat_prob
        self.T = mat_prob.shape[0]
        self.F = mat_prob.shape[1]
        self.score_func = Score(tok, self.mat_prob)

    def init_paths(self):
        init_score_b, init_score_nb = dict(), dict()
        init_paths_b, init_paths_nb = set(), set()
        # if path blank
        path = ''
        init_score_b[self.blank] = self.score_func(0,self.blank)
        init_paths_b = {path}
        # all except blank
        for c in self.symbols[:-1]:
            path = c
            init_score_nb[path] = self.score_func(0,c)
            init_paths_nb.add(path)
        return(init_paths_b, init_paths_nb, init_score_b, init_score_nb)
   
    def extend_with_blank(self, paths_b, paths_nb,t):
        """
        extends all paths with blanks and output updated path&score for blanks
        """
        new_paths_b = set()
        new_scores_b = dict()
        # paths ending with blank
        for path in paths_b:
            if path == '':
                new_paths_b.add(path)
                new_scores_b[self.blank] = self.score_b[self.blank] * self.score_func(t,self.blank)
            else:
                new_paths_b.add(path)
                new_scores_b[path] = self.score_b[path] * self.score_func(t,self.blank)
        
        # symbol path are extended with blanks so belong to paths_b
        for path in paths_nb:
            # if path already in new_path_b
            if path in new_paths_b:
                new_scores_b[path] += self.score_nb[path] * self.score_func(t,self.blank)
            else:
                new_paths_b.add(path)
                new_scores_b[path] = self.score_nb[path] * self.score_func(t,self.blank)
        return(new_paths_b, new_scores_b)

    def extend_with_symbol(self, paths_b, paths_nb, t):
        """
        extends all paths with symbols and update paths with symbols
        """
        new_paths_nb = set()
        new_scores_nb = dict()
        # extend paths ending with blank

        for path in paths_b:
            for c in self.symbols[:-1]:
                new_path = path + c
                new_paths_nb.add(new_path)
                if path == '':
                    new_scores_nb[new_path] = self.score_b[self.blank] * self.score_func(t,c)
                else:
                    new_scores_nb[new_path] = self.score_b[path] * self.score_func(t,c)

        for path in paths_nb:
            for c in self.symbols[:-1]:
                # if new symbol = last symbol don't update path
                if c == path[-1]:
                    new_path = path
                else:
                    new_path = path + c
                if new_path in new_paths_nb:
                    new_scores_nb[new_path] += self.score_nb[path] * self.score_func(t,c)
                else:
                    new_paths_nb.add(new_path)
                    new_scores_nb[new_path] = self.score_nb[path] * self.score_func(t,c)
            
        return(new_paths_nb, new_scores_nb)

    def prune_paths(self, paths_b, paths_nb, scores_b, scores_nb, beam_width):
        """
        keep beam_width best paths and store in global b, bs scores
        """
        pruned_scores_b = dict()
        pruned_scores_nb = dict()
        pruned_paths_b = set()
        pruned_paths_nb = set()

        blank = dict()
        all = []
        for item in [(v,k) for (k,v) in scores_b.items()]:
            blank[item] = True
            all.append(item)
        for item in [(v,k) for (k,v) in scores_nb.items()]:
            blank[item] = False
            all.append(item)
        beams = sorted(all)[-beam_width:]
        for b in beams:
            if blank[b] == True:
                if b[1] == self.blank:
                    pruned_paths_b.add('')
                    pruned_scores_b[self.blank] = b[0]
                else:
                    pruned_paths_b.add(b[1])
                    pruned_scores_b[b[1]] = b[0]
            else:
                if b[1] == self.blank:
                    pruned_paths_nb.add('')
                    pruned_scores_nb[self.blank] = b[0]
                else:
                    pruned_paths_nb.add(b[1])
                    pruned_scores_nb[b[1]] = b[0]

        self.score_b, self.score_nb = pruned_scores_b, pruned_scores_nb

        return(pruned_paths_b, pruned_paths_nb)

    def merge_paths(self, paths_b, paths_nb, scores_b, scores_nb):
        """
        merge same paths and updated scores
        """
        merged_paths = paths_nb
        final_score = scores_nb
        for path in paths_b:
            if path in merged_paths:
                if path == '':
                    final_score[self.blank] += scores_b[self.blank]
                else:
                    final_score[path] += scores_b[path]
            else:
                merged_paths.add(path)
                if path == '':
                    final_score[self.blank] = scores_b[self.blank]
                else:
                    final_score[path] = scores_b[path]

        return(merged_paths, final_score)

    def decode(self, beam_width):
        """
        init, extend paths with blanks, extend paths with symbol, merge resulting path and rank
        """
        new_paths_b, new_paths_nb, new_scores_b, new_scores_nb = self.init_paths()

        for i in range(1, self.T):
            paths_b, paths_nb = \
                self.prune_paths(new_paths_b, new_paths_nb, new_scores_b, new_scores_nb, beam_width)
            new_paths_b, new_scores_b = self.extend_with_blank(paths_b, paths_nb, i)
            new_paths_nb, new_scores_nb = self.extend_with_symbol(paths_b, paths_nb, i)

        merged_paths, final_score = self.merge_paths(new_paths_b, new_paths_nb,new_scores_b, new_scores_nb)
        return(max(final_score, key=final_score.get))


if __name__ == "__main__":
    # syms = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    syms = 'ab-'
    t = Tokenizer(syms)
    print(t)
    print(t.char2int('b'), t.int2char(1))
    #                     a      b    -
    mat_prob = np.array([[0.2, 0.0, 0.8],
                        [0.4, 0.0, 0.6]])

    score = Score(t, mat_prob)
    print(score(1,'-'))
    bs = BeamSearch(syms, mat_prob)
    bs.decode(2)