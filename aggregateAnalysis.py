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
    parser.add_argument("-d", "--dataname", nargs="+", type=str, help="dataset name", default=["fb15k", "wn18"])
    parser.add_argument("-t", "--type", nargs="+", type=str, help="vector type [ent/rel]", default=["ent", "rel"])
    parser.add_argument("-g", "--geometry", nargs="+", type=str, help="geometry feature[length/conicity]", default=["conicity", "length"])
    parser.add_argument("-o", "--opdir", type=str, help="output directory", required=True)
    parser.add_argument("-i", "--indir", type=str, help="input directory containing geometry pickle files", required=True)
    #parser.add_argument("--result", dest="result", help="true for plotting existing results", action="store_true")
    #parser.set_defaults(result=False)
    #parser.set_defaults(pr=False)
    return parser

def avgNegative(indir, typ, dataname, dim, methods):
    infile = os.path.join(indir, "%s.%s.d%d"%(typ, dataname, dim))
    with open(infile+".p", "rb") as fin:
        result = pickle.load(fin)
    mean_products = result['mean_products']
    avg_products = {}
    for nneg, vals in mean_products.iteritems():
        cur_sum = 0
        nz_len = 0
        for method in methods:
            if sum(abs(vals[method])) > 0:
                cur_sum = cur_sum+vals[method]
                nz_len += 1
        cur_sum = cur_sum/nz_len
        avg_products[nneg] = cur_sum
    return avg_products

def avgDimension(indir, typ, dataname, nneg, methods):
    infile = os.path.join(indir, "%s.%s.n%d"%(typ, dataname, nneg))
    with open(infile+".p", "rb") as fin:
        result = pickle.load(fin)
    mean_products = result['mean_products']
    avg_products = {}
    for dim, vals in mean_products.iteritems():
        cur_sum = 0
        nz_len = 0
        for method in methods:
            if sum(abs(vals[method])) > 0:
                cur_sum = cur_sum+vals[method]
                nz_len += 1
        cur_sum = cur_sum/nz_len
        avg_products[dim] = cur_sum
    return avg_products

def aggregateAnalysis(args):
    #self.cfg = ConfigParser()
    #self.cfg.read(args.cfgFile)
    methods = ['transe', 'transr', 'stranse', 'distmult', 'hole', 'complex']
    add_models = ['transe', 'transr', 'stranse']
    mult_models = ['distmult', 'hole', 'complex']
    #nnegs  = [1, 50, 100]
    #dims = [50, 100]
    nnegs  = [1]
    dims = [100]
    useEnt = True
    for typ in args.type:
        for geometry in args.geometry:
                for dim in dims:
                    bar_heights = {}
                    outputfile = os.path.join(args.opdir, geometry, "%s.d%d"%(typ, dim))
                    for dataname in args.dataname:
                        negDir = os.path.join(args.indir, "negativeAnalysis", geometry)
                        avg_add = avgNegative(negDir, typ, dataname, dim, add_models)
                        avg_mult= avgNegative(negDir, typ, dataname, dim, mult_models)
                        cur_dict = bar_heights.setdefault(dataname, {})
                        for nneg, vals in avg_add.iteritems():
                            #bar_heights.setdefault(nneg, []).extend([vals[-1], avg_mult[nneg][-1]])
                            cur_dict[nneg] = {"add":vals[-1], "mult":avg_mult[nneg][-1]}
                    bar_height_list = []
                    xticks = sorted(bar_heights[dataname].keys())
                    for nneg in xticks:
                        cur_heights = []
                        legends = []
                        for mtype in ['add', 'mult']:
                            for dataname in args.dataname:
                                cur_heights.append(bar_heights[dataname][nneg][mtype])
                                legends.append((dataname, mtype))
                        bar_height_list.append(cur_heights)
                    #legends = [legend for legend in product(args.dataname, ["add", "mult"])]
                    plotBars(bar_height_list, xlabel="#NegativeSamples", ylabel="Average %s"%geometry, legends=legends, xticks=xticks, outfile=outputfile, show=False)

                for nneg in nnegs:
                    bar_heights = {}
                    outputfile = os.path.join(args.opdir, geometry, "%s.n%d"%(typ, nneg))
                    for dataname in args.dataname:
                        dimDir = os.path.join(args.indir, "dimensionAnalysis", geometry)
                        avg_add = avgDimension(dimDir, typ, dataname, nneg, add_models)
                        avg_mult= avgDimension(dimDir, typ, dataname, nneg, mult_models)
                        cur_dict = bar_heights.setdefault(dataname, {})
                        for dim, vals in avg_add.iteritems():
                            #bar_heights.setdefault(dim, []).extend([vals[-1], avg_mult[dim][-1]])
                            cur_dict[dim] = {"add":vals[-1], "mult":avg_mult[dim][-1]}
                    bar_height_list = []
                    xticks = sorted(bar_heights[dataname].keys())
                    for dim in xticks:
                        cur_heights = []
                        legends = []
                        for mtype in ['add', 'mult']:
                            for dataname in args.dataname:
                                cur_heights.append(bar_heights[dataname][dim][mtype])
                                legends.append((dataname, mtype))
                        bar_height_list.append(cur_heights)
                    """
                    for dim in xticks:
                        bar_height_list.append(bar_heights[dim])
                    """
                    #legends = [legend for legend in product(args.dataname, ["add", "mult"])]
                    plotBars(bar_height_list, xlabel="#Dimensions", ylabel="Average %s"%geometry, legends=legends, xticks=xticks, outfile=outputfile, show=False)
    """
    for dim in dims:
        if not args.result:
            mean_products = {}
            mean_products_list = []
            for nneg in nnegs:
                cur_mean_products_list= []
                for method in methods:
                    modelfile = "%s.%s.n%d.d%d.p" %(args.dataname, method, nneg, dim)
                    modelfile = os.path.join(args.mdir, modelfile)
                    if not os.path.exists(modelfile):
                        print modelfile
                        mean_products.setdefault(nneg, {})[method] = np.array([0, 0, 0, 0, 0], dtype=np.float32)
                        cur_mean_products_list.append(np.float32(0.0))
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
                    mean_products.setdefault(nneg, {})[method] = np.array(mgp, dtype=np.float32)
                    cur_mean_products_list.append(np.float32(mgp[-1]))
                mean_products_list.append(cur_mean_products_list)
            outputfile = os.path.join(args.opdir, args.geometry, "%s.%s.d%d"%(args.type, args.dataname, dim))
            #plotBars(mean_products_list, xlabel="#negatives", ylabel="Avg MeanProduct", legends=methods, xticks=nnegs, outfile=outputfile, show=False)
            with open(outputfile+".p", "wb") as fout:
                pickle.dump({"mean_products":mean_products, "mean_products_list":mean_products_list, "methods":methods, "nnegs":nnegs, "dim":dim}, fout)
        else:
            outputfile = os.path.join(args.opdir, args.geometry, "%s.%s.d%d"%(args.type, args.dataname, dim))
            with open(outputfile+".p", "rb") as fin:
                """ """
                mean_products = pickle.load(fin)
                mean_products_list = []
                for nneg in nnegs:
                    cur_products_list = []
                    for method in methods:
                        cur_products_list.append(np.float32(mean_products[nneg][method][-1]))
                    mean_products_list.append(cur_products_list)
                """ """
                result = pickle.load(fin)
                mean_products_list = result['mean_products_list']
                if args.geometry in ['length']:
                    ylabel = 'length'
                else:
                    ylabel = 'conicity'
                plotBars(mean_products_list, xlabel="#NegativeSamples", ylabel=ylabel, legends=methods, xticks=nnegs, outfile=outputfile, show=False)
                """

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    aggregateAnalysis(args)
    #avgNegative(args.indir, "ent", "fb15k", 100, ['transe', 'transr', 'stranse'])

if __name__ == "__main__":
	main()
