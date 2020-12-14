#!/usr/bin/env python
# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>


class CTCDecoder(object):
    def __init__(self, classes, blank_index=0):
        self.classes = classes
    
    def decode(self,probs_mat):
        pass        


class GreedyDecoder(CTCDecoder):
    def __init__(self):
        super.__init__(classes)

if __name__ == "__main__":
    pass    
