#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

from typing import List, Set, Dict, Tuple, Optional, Callable
import string
from queue import Queue
import torch


class CharacterTokenizer:
    """
    Convert sentences into indices of characters and back
    """
    def __init__(self, classes:str=string.ascii_lowercase):
        m = {}
        # TODO(slg): <EOS>?
        counter = 0
        for letter in classes:
            m[letter] = counter
            counter += 1
        # m["'"] = 0
        special_tokens = ['<SPACE>', '<PAD>', '<BOS>', '<EOS>']
        if classes == string.ascii_lowercase:
            for token in special_tokens:
                m[token] = len(m)
        self._map = m
        self._pam = self.inverse_map(self.map)

    def __len__(self)->Dict[str,int]:
        return(len(self._map))
    
    @staticmethod
    def inverse_map(map:Dict[str, int])->Dict[int,str]:
        return(dict([ (v,k) for (k,v) in map.items()]))
    
    @property
    def map(self)->Dict[str,int]:
        return(self._map)

    @property
    def vocab(self):
        return(list(self._map.keys()))

    @property
    def pam(self)->Dict[int,str]:
        return(self._pam)
    
    def text2int(self, text:str)->List[int]:
        ints = []
        for c in text.lower():
            if c == " ":
                i = self._map['<SPACE>']
            else:
                i = self._map[c]
            ints.append(i)
        return(ints)

    def int2text(self, ints:List[int])->List[str]:
        chars = []
        for i in ints:
            c = self._pam[i]
            chars.append(c)
        return(''.join(chars).replace("<SPACE>", ' '))


class CTCDecoder(object):
    """
    Base Decoder class that provides decoding of string of ctc characters (with blank) into string of text
    """
    def __init__(self, char_tok:CharacterTokenizer, blank_index:int=0):
        self.tok = char_tok
        self.blank_index = blank_index

    def charseq_decode(self, sequence:List[int])->str:
        res = []
        sequence = [ int(x) for x in sequence ]
        for i, v in enumerate(sequence):
            if i==0:
                res.append(self.tok.pam[v])
            else:
                if (v != sequence[i-1]) and (v != self.blank_index):
                    res.append(self.tok.pam[v])
        return(''.join(res))
    
    def decode(self, probs_mat:torch.Tensor):
        raise NotImplementedError


class Test(CTCDecoder):
    """
    try stuff
    """
    def __init__(self, char_tok, blank_index=0):
        super().__init__(char_tok, blank_index)
        
    
    def decode(self, probs:torch.Tensor)->str:
        pass        


class GreedyDecoder(CTCDecoder):
    """
    Decoder that chooses character with higher probability at each time step (best path)
    """
    def __init__(self, char_tok:CharacterTokenizer, blank_index:int=0):
        super().__init__(char_tok, blank_index)

<<<<<<< HEAD
    def decode(self, probs:torch.Tensor)->str:
        best_path = torch.argmax(probs, dim=0)
        return(self.charseq_decode(best_path))
=======
    def decode(self, probs):
        best_paths = torch.argmax(probs, dim=2)
        return([self.charseq_decode(x) for x in best_paths])


class Beam():
    def __init__(self, seq='', prob_b=1.0, prob_nb=0.0):
        self.seq = seq
        self.prob_b = prob_b
        self.prob_nb = prob_nb
    
    def extend(self, c, prob_c):
        if c == '<BLANK>':
            self.prob_b += prob_c *( self.prob_b + self.prob_nb)



class BeamSearchDecoder(CTCDecoder):
    def __init__(self, char_tok, blank_index=0):
        super().__init__(char_tok, blank_index)

    def decode(mat_probs, beam_size):
        n_t = mat_probs.shape[1]
        n_c = mat_probs.shape[2]
        empty_beam = Beam()
        beams = [empty_beam]
        for i in range(n_t):
            
            

>>>>>>> 4f146832562266cc949a63345fef1e2bf8fe8098



class SimpleBeamSearch(CTCDecoder):
    """
    Decoder that keeps track of N best beams at each time step
    """
    def __init__(self, char_tok, blank_index=0):
        super().__init__(char_tok, blank_index)
         
    
    def decode(self, probs:torch.Tensor)->str:
        pass




if __name__ == "__main__":
    pass    
