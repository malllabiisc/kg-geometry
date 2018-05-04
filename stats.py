import sys
import os

import numpy as np
import argparse
import cPickle as pickle

from triples import Triples
class Stats:
    def __init__(self, t, usePR=False):
        self.t = t
        self.setStats(usePR)
        np.random.seed(123123)

    def setStats(self, usePR):
        ne = len(self.t.eNames)
        nr = len(self.t.rNames)
        self.eFreq = np.zeros((ne, ), dtype='int32')
        self.rFreq = np.zeros((nr, ), dtype='int32')
        for s, o, p in self.t.train:
            self.eFreq[s] += 1
            self.eFreq[o] += 1
            self.rFreq[p] += 1
        if usePR:
            print "Using pagerank"
            self.eIndices = (-self.t.pagerank).argsort()
        else:
            self.eIndices = (-self.eFreq).argsort()
        self.rIndices = (-self.rFreq).argsort()

    def getEnts(self, rankBand, nSamples):
        if rankBand[1] < 0:
            x = np.arange(self.eIndices.shape[0]-rankBand[0])
        else:
            x = np.arange(rankBand[1]-rankBand[0])
        np.random.shuffle(x)
        x = x[:nSamples] + rankBand[0]
        return self.eIndices[x]

    def getRels(self, rankBand, nSamples):
        if rankBand[1] < 0:
            x = np.arange(self.rIndices.shape[0]-rankBand[0])
        else:
            x = np.arange(rankBand[1]-rankBand[0])
        np.random.shuffle(x)
        x = x[:nSamples] + rankBand[0]
        return self.rIndices[x]

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments")
    parser.add_argument("-d", "--datafile", type=str, help="pickled triple file", required=True)
    parser.add_argument("-o", "--outfile", type=str, help="file to save stats")

    return parser

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    t = Triples(args.datafile)
    stats = Stats(t)
    rRanges = [((0,50), 50), ((50,100), 50), ((100,200), 100), ((200, 500), 300), ((500,t.nr), t.nr-500)]
    idxSets = []
    for rankBand, ns in rRanges:
        idxSets.append(stats.getRels(rankBand, ns))
    rels = []
    cats = []
    for idxSet in idxSets:
        cur_rels = []
        cur_cats = {}
        for idx in idxSet:
            cur_rels.append(t.rNames[idx])
            cat = t.rNames[idx].split("/")[1]
            cur_cats[cat] = cur_cats.get(cat,0)+1
        rels.append(cur_rels)
        cats.append(cur_cats)
    if args.outfile:
        with open(args.outfile+'.rel.txt', "w") as fout:
            for idx, cur_rels in enumerate(rels):
                fout.write("%s\n"%str(rRanges[idx][0]))
                for rel in cur_rels:
                    fout.write("%s\n"%rel)
        with open(args.outfile+'.cat.txt', "w") as fout:
            for idx, cur_cats in enumerate(cats):
                fout.write("%s\n"%str(rRanges[idx][0]))
                for cat, count in cur_cats.iteritems():
                    fout.write("%s:%d\t"%(cat, count))
                fout.write("\n")

    import pdb; pdb.set_trace()



if __name__ == "__main__":
	main()
