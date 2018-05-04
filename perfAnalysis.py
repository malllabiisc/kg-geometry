import sys
import os

import argparse
import cPickle as pickle
from ConfigParser import ConfigParser as ConfigParser
from itertools import product

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.manifold import TSNE
import scipy.stats as scistats

from stats import Stats
from model import Model
from triples import Triples
from util import *
from analysis import Analyser
from typeAnalysis import best_methods, uniform_methods

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mdir", type=str, help="directory containing the models", default="./data")
    parser.add_argument("-d", "--dataname", type=str, help="dataset name", default="fb15k")
    parser.add_argument("-t", "--type", type=str, help="vector type [ent/rel]", default="ent")
    parser.add_argument("-g", "--geometry", type=str, help="geometry feature[length/conicity]", required=True)
    parser.add_argument("-p", "--perffile", type=str, help="files containing model performances", required=True)
    parser.add_argument("-o", "--opdir", type=str, help="output directory", required=True)
    parser.add_argument("--result", dest="result", help="true for plotting existing results", action="store_true")
    parser.set_defaults(result=False)
    #parser.add_argument("-d", "--datafile", type=str, help="pickled triples file", required=True)
    #parser.add_argument("-m", "--modelfile", type=str, help="pickled model file", required=True)
    #parser.add_argument("-c", "--cfgfile", type=str, help="config file containing list of model and data files", default="./exp.cfg")
    #parser.add_argument("--pr", dest="pr", help="Flag for using pagerank plot", action='store_true')
    #parser.set_defaults(pr=False)
    return parser

def readPerfs(filename):
    perfs = {}
    delimiter = "\t"
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                x = line.split(delimiter)
                dim = int(x[1])
                nneg = int(x[2])
                method = x[0].lower()
                hits_10 = np.float32(x[5])
                if hits_10 < 1:
                    hits_10 = 100*hits_10
                perf = {"mr":np.float32(x[3]), "mrr":np.float32(x[4]), "hits_10":hits_10}
                perfs.setdefault(dim, {}).setdefault(nneg, {})[method] = perf
    return perfs

def perfAnalysis(args):
    #self.cfg = ConfigParser()
    #self.cfg.read(args.cfgFile)
    methods = ['transe', 'transr', 'stranse', 'distmult', 'hole', 'complex']
    nnegs  = [1, 50, 100]
    dims = [50, 100]
    mean_products = {}
    name_conicity = {}
    useEnt = True
    if not args.result:
        for dim in dims:
            for nneg in nnegs:
                for method in methods:
                    modelfile = "%s.%s.n%d.d%d.p" %(args.dataname, method, nneg, dim)
                    modelfile = os.path.join(args.mdir, modelfile)
                    datafile = "%s.%s.bin" % (args.dataname, method)
                    datafile = os.path.join(args.mdir, datafile)
                    analyser = Analyser(datafile, modelfile, usePR=False)
                    #nSamples = 100
                    #eRanges = [((0,100), nSamples), ((100,500), nSamples), ((500,5000), nSamples), ((5000, analyser.t.ne), nSamples)]
                    #entIndices = analyser.getEntIdxs(eRanges)
                    if args.type in ['ent']:
                        nSamples = 100
                        ranges = [((0,100), nSamples), ((100,500), nSamples), ((500,5000), nSamples), ((5000, analyser.t.ne), nSamples)]
                        indices = analyser.getEntIdxs(ranges)
                        useEnt = True
                    else:
                        nSamples = 100
                        if args.dataname in ['wn18']:
                            ranges = [((0,3), 3), ((3,10), 7), ((10,analyser.t.nr), analyser.t.nr-10)]
                        else:
                            ranges = [((0,100), nSamples), ((100,500), nSamples), ((500,analyser.t.nr), nSamples)]
                        indices = analyser.getRelIdxs(ranges)
                        useEnt = False
                    legendLabels=[]
                    for a,b in ranges:
                        curLabel = "%d-%d"%(a[0],a[1])
                        legendLabels.append(curLabel)
                    if args.geometry in ['length']:
                        gp, mgp = analyser.getLengths(indices, ent=useEnt)
                    else:
                        gp, mgp = analyser.getInnerProducts(indices, sampleMean=True, ent=useEnt, normalized=True)
                    print "%s\tneg %d" % (method,nneg)
                    print mgp
                    mean_products.setdefault(dim, {}).setdefault(nneg, {})[method] = np.array(mgp, dtype=np.float32)
                    mname = "%s.%s.n%d.d%d" % (args.dataname, method, nneg, dim)
                    name_conicity[mname] = mgp[-1]
        outputfile = os.path.join(args.opdir, args.geometry, "%s.%s"%(args.type, args.dataname))
        with open(outputfile+".p", "wb") as fout:
            pickle.dump({"mean_products":mean_products, "methods":methods, "nnegs":nnegs, "dims":dims, "name_conicity":name_conicity}, fout)
            #pickle.dump({"mean_products":mean_products, "mean_products_list":mean_products_list, "methods":methods, "nnegs":nnegs, "dim":dim}, fout)
    else:
        outputfile = os.path.join(args.opdir, args.geometry, "%s.%s"%(args.type, args.dataname))
        with open(outputfile+".p", "rb") as fin:
            result = pickle.load(fin)
        if "perfs" not in result:
            with open(args.perffile, "rb") as fin:
                """
                mean_products = pickle.load(fin)
                mean_products_list = []
                for nneg in nnegs:
                    cur_products_list = []
                    for method in methods:
                        cur_products_list.append(np.float32(mean_products[nneg][method][-1]))
                    mean_products_list.append(cur_products_list)
                """
                result['perfs'] = pickle.load(fin)
        #perfs = readPerfs(args.perffile)
        whitelist = []
        for method, nneg, dim in product(result['methods'], result['nnegs'], result['dims']):
            if dim == 100:
                if method in ['hole', 'complex', 'distmult']:
                    whitelist.append("%s.n%d.d%d"%(method, nneg, dim))
                elif method in ['transe', 'stranse'] and nneg in [1]:
                    whitelist.append("%s.n%d.d%d"%(method, nneg, dim))
                elif method in ['transr']:
                    if nneg == 1:
                        whitelist.append("%s.n%d.d%d"%(method, nneg, dim))
                    elif dim == 100:
                        whitelist.append("%s.n%d.d%d"%(method, nneg, dim))
        if args.geometry in ['length']:
            plotConePerf(methods, nnegs, dims, result, outputfile, xlabel="length", whitelist=whitelist, show=True)
        else:
            plotConePerf(methods, nnegs, dims, result, outputfile, xlabel="conicity", whitelist=whitelist, show=True)

       
def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    perfAnalysis(args)

if __name__ == "__main__":
	main()
