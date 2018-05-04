import sys
import os

import argparse
import cPickle as pickle
import numpy as np

MIN_PR = 1e-10
class Triples:
    def __init__(self, filename):
        x = pickle.load(open(filename, 'rb'))
        self.train = x['train_subs']
        self.test = x['test_subs']
        self.valid = x['valid_subs']
        self.eNames = x['entities']
        self.rNames = x['relations']
        self.ne = len(self.eNames)
        self.nr = len(self.rNames)
        self.pagerank = np.zeros((self.ne,), dtype=np.float32)
        if x.get("pagerank", None) is not None:
            for idx, ent in enumerate(self.eNames):
                #self.pagerank.append(x['pagerank'].get(ent, MIN_PR))
                self.pagerank[idx] = x['pagerank'].get(ent, MIN_PR)
    
    def groupByRelation(self, sub):
        # format is (head, tail, relations)
        triples = self.__dict__.get(sub)
        import pdb; pdb.set_trace()
        if triples is None:
            return {}
        rel_triples = {}
        for h,t,r in triples:
            rel_triples.setdefault(r, []).append((h,t))
        return rel_triples


def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments")
    parser.add_argument("-f", "--filename", type=str, help="pickled data file", required=True)
    return parser

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    t = Triples(args.filename)

if __name__ == "__main__":
	main()

