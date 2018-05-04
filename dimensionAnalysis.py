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

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mdir", type=str, help="directory containing the models", default="./data")
    parser.add_argument("-d", "--dataname", type=str, help="dataset name", default="fb15k")
    parser.add_argument("-t", "--type", type=str, help="vector type [ent/rel]", default="ent")
    parser.add_argument("-g", "--geometry", type=str, help="geometry feature[length/conicity]", required=True)
    parser.add_argument("-o", "--opdir", type=str, help="output directory", required=True)
    parser.add_argument("--result", dest="result", help="true for plotting existing results", action="store_true")
    parser.set_defaults(result=False)
    #parser.add_argument("-d", "--datafile", type=str, help="pickled triples file", required=True)
    #parser.add_argument("-m", "--modelfile", type=str, help="pickled model file", required=True)
    #parser.add_argument("-c", "--cfgfile", type=str, help="config file containing list of model and data files", default="./exp.cfg")
    #parser.add_argument("--pr", dest="pr", help="Flag for using pagerank plot", action='store_true')
    #parser.set_defaults(pr=False)
    return parser

def dimAnalysis(args):
    #self.cfg = ConfigParser()
    #self.cfg.read(args.cfgFile)
    methods = ['transe', 'transr', 'stranse', 'distmult', 'hole', 'complex']
    nnegs  = [1]
    dims = [50, 100, 200]
    nBins = 0
    useEnt = True
    for nneg in nnegs:
        if not args.result:
            mean_products = {}
            mean_products_list = []
            for dim in dims:
                cur_mean_products_list= []
                for method in methods:
                    modelfile = "%s.%s.n%d.d%d.p" %(args.dataname, method, nneg, dim)
                    modelfile = os.path.join(args.mdir, modelfile)
                    if not os.path.exists(modelfile):
                        print "file not found ... ", modelfile
                        if args.type in ['ent']:
                            nBins = 5
                        else:
                            nBins = 4
                        mean_products.setdefault(dim, {})[method] = np.zeros((nBins,), dtype=np.float32)
                        cur_mean_products_list.append(0.0)
                        continue
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
                            ranges = [((0,3), 3), ((3,10), 3), ((10,analyser.t.nr), 3)]
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
                    nBins = len(mgp)
                    print "%s\tdim %d" % (method,dim)
                    print mgp
                    mean_products.setdefault(dim, {})[method] = np.array(mgp, dtype=np.float32)
                    cur_mean_products_list.append(np.float32(mgp[-1]))
                mean_products_list.append(cur_mean_products_list)
            outputfile = os.path.join(args.opdir, args.geometry, "%s.%s.n%d"%(args.type, args.dataname, nneg))
            #plotBars(mean_products_list, xlabel="#dimensions", ylabel="Avg MeanProduct", legends=methods, xticks=dims, outfile=outputfile, show=False)
            with open(outputfile+".p", "wb") as fout:
                pickle.dump({"mean_products":mean_products, "mean_products_list":mean_products_list, "methods":methods, "nneg":nneg, "dims":dims}, fout)
        else:
            outputfile = os.path.join(args.opdir, args.geometry, "%s.%s.n%d"%(args.type, args.dataname, nneg))
            with open(outputfile+".p", "rb") as fin:
                """
                mean_products = pickle.load(fin)
                mean_products_list = []
                for dim in dims:
                    cur_products_list = []
                    for method in methods:
                        cur_products_list.append(np.float32(mean_products[dim][method][-1]))
                    mean_products_list.append(cur_products_list)
                """
                result = pickle.load(fin)
                mean_products_list = result['mean_products_list']
                if args.geometry in ['length']:
                    ylabel = 'length'
                else:
                    ylabel = 'conicity'
                plotBars(mean_products_list, xlabel="#Dimensions", ylabel=ylabel, legends=methods, xticks=dims, outfile=outputfile, show=False)

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    dimAnalysis(args)

if __name__ == "__main__":
	main()
