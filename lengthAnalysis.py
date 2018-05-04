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
    parser.add_argument("-o", "--opdir", type=str, help="output directory", required=True)
    parser.add_argument("--result", dest="result", help="true for plotting existing results", action="store_true")
    parser.set_defaults(result=False)
    #parser.add_argument("-d", "--datafile", type=str, help="pickled triples file", required=True)
    #parser.add_argument("-m", "--modelfile", type=str, help="pickled model file", required=True)
    #parser.add_argument("-c", "--cfgfile", type=str, help="config file containing list of model and data files", default="./exp.cfg")
    #parser.add_argument("--pr", dest="pr", help="Flag for using pagerank plot", action='store_true')
    #parser.set_defaults(pr=False)
    return parser

def negAnalysis(args):
    #self.cfg = ConfigParser()
    #self.cfg.read(args.cfgFile)
    methods = ['transe', 'transr', 'stranse', 'distmult', 'hole', 'complex']
    nnegs  = [1, 50, 100]
    dims = [50, 100]
    for dim in dims:
        if not args.result:
            mean_products = {}
            mean_products_list = []
            for nneg in nnegs:
                cur_mean_products_list= []
                for method in methods:
                    modelfile = "%s.%s.n%d.d%d.p" %(args.dataname, method, nneg, dim)
                    modelfile = os.path.join(args.mdir, modelfile)
                    datafile = "%s.%s.bin" % (args.dataname, method)
                    datafile = os.path.join(args.mdir, datafile)
                    analyser = Analyser(datafile, modelfile, usePR=False)
                    nSamples = 100
                    eRanges = [((0,100), nSamples), ((100,500), nSamples), ((500,5000), nSamples), ((5000, analyser.t.ne), nSamples)]
                    entIndices = analyser.getEntIdxs(eRanges)
                    legendLabels=[]
                    for a,b in eRanges:
                        curLabel = "%d-%d"%(a[0],a[1])
                        legendLabels.append(curLabel)
                    gp, mgp = analyser.getInnerProducts(entIndices, sampleMean=False, ent=True, normalized=True)
                    print "%s\tneg %d" % (method,nneg)
                    print mgp
                    mean_products.setdefault(nneg, {})[method] = np.array(mgp, dtype=np.float32)
                    cur_mean_products_list.append(np.float32(mgp[-1]))
                mean_products_list.append(cur_mean_products_list)
            outputfile = os.path.join(args.opdir, "%s.d%d"%(args.dataname, dim))
            plotBars(mean_products_list, xlabel="#negatives", ylabel="Avg MeanProduct", legends=methods, xticks=nnegs, outfile=outputfile, show=False)
            with open(outputfile+".p", "wb") as fout:
                pickle.dump({"mean_products":mean_products, "mean_products_list":mean_products_list, "methods":methods, "nnegs":nnegs, "dim":dim}, fout)
        else:
            outputfile = os.path.join(args.opdir, "%s.d%d"%(args.dataname, dim))
            with open(outputfile+".p", "rb") as fin:
                """
                mean_products = pickle.load(fin)
                mean_products_list = []
                for nneg in nnegs:
                    cur_products_list = []
                    for method in methods:
                        cur_products_list.append(np.float32(mean_products[nneg][method][-1]))
                    mean_products_list.append(cur_products_list)
                """
                result = pickle.load(fin)
                mean_products_list = result['mean_products_list']
                plotBars(mean_products_list, xlabel="#negatives", ylabel="Avg MeanProduct", legends=methods, xticks=nnegs, outfile=outputfile, show=False)


    #mean_products_list = [[-0.06280816346406937, -0.0030675753951072693, -0.016776302829384804, 0.33246153593063354, 0.44574713494461105, 0.2805495858192444], [0.00207449309527874, -0.004043783526867628, -0.015670031309127808, 0.46322575211524963, 0.8434605554397576, 0.43073755502700806], [-0.028388632461428642, -0.002486256416887045, -0.016531826928257942, 0.5173962712287903, 0.8887729744164399, 0.4763573408126831]]


def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    negAnalysis(args)

if __name__ == "__main__":
	main()
