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

best_methods = {"fb15k":{"transe" : {"nneg":1, "dim":200},
                         "transr" : {"nneg":1, "dim":200},
                         "stranse": {"nneg":1, "dim":200},
                         "distmult":{"nneg":50,"dim":200},
                         "hole"   : {"nneg":50,"dim":200},
                         "complex": {"nneg":50,"dim":100},
                        },
                "wn18" :{"transe" : {"nneg":50,"dim":100},
                         "transr" : {"nneg":1, "dim":50 },
                         "stranse": {"nneg":1, "dim":50 },
                         "distmult":{"nneg":1 ,"dim":200},
                         "hole"   : {"nneg":1 ,"dim":150},
                         "complex": {"nneg":1 ,"dim":150},
                        }
                }

uniform_methods = {"fb15k":{"transe" : {"nneg":1, "dim":100},
                         "transr" : {"nneg":1, "dim":100},
                         "stranse": {"nneg":1, "dim":100},
                         "distmult":{"nneg":1, "dim":100},
                         "hole"   : {"nneg":1, "dim":100},
                         "complex": {"nneg":1, "dim":100},
                        },
                "wn18" :{"transe" : {"nneg":1, "dim":100},
                         "transr" : {"nneg":1, "dim":100},
                         "stranse": {"nneg":1, "dim":100},
                         "distmult":{"nneg":1, "dim":100},
                         "hole"   : {"nneg":1, "dim":100},
                         "complex": {"nneg":1, "dim":100},
                        }
                }

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mdir", type=str, help="directory containing the models", default="./data")
    parser.add_argument("-d", "--dataname", type=str, help="dataset name", default="fb15k")
    parser.add_argument("-t", "--type", type=str, help="vector type [ent/rel]", default="ent")
    parser.add_argument("-g", "--geometry", type=str, help="geometry feature[length/conicity]", required=True)
    parser.add_argument("-o", "--opdir", type=str, help="output directory", required=True)
    parser.add_argument("--result", dest="result", help="true for plotting existing results", action="store_true")
    parser.set_defaults(result=False)
    parser.add_argument("--nocombined", dest="combined", help="true for merging frequency bands", action="store_false")
    parser.set_defaults(combined=True)
    parser.add_argument("--best", dest="best", help="true for plotting best settings results", action="store_true")
    parser.set_defaults(best=False)
    #parser.add_argument("-d", "--datafile", type=str, help="pickled triples file", required=True)
    #parser.add_argument("-m", "--modelfile", type=str, help="pickled model file", required=True)
    #parser.add_argument("-c", "--cfgfile", type=str, help="config file containing list of model and data files", default="./exp.cfg")
    #parser.add_argument("--pr", dest="pr", help="Flag for using pagerank plot", action='store_true')
    #parser.set_defaults(pr=False)
    return parser


def typeAnalysis(args):
    #self.cfg = ConfigParser()
    #self.cfg.read(args.cfgFile)
    #methods = ['transe', 'transr', 'stranse', 'distmult', 'hole', 'complex']
    #nnegs  = [1, 50, 100]
    #dims = [50, 100]
    if args.best:
	methods = best_methods
	args.opdir = os.path.join(args.opdir, "best")
    else:
	methods = uniform_methods
    useEnt = True
    markers = "+.x3ov^<>p"
    if not args.result:
        mean_products = {}
        for method, vals in methods[args.dataname].iteritems():
            nneg = vals['nneg']
            dim = vals['dim']
            modelfile = "%s.%s.n%d.d%d.p" %(args.dataname, method, nneg, dim)
            modelfile = os.path.join(args.mdir, modelfile)
            datafile = "%s.%s.bin" % (args.dataname, method)
            datafile = os.path.join(args.mdir, datafile)
            analyser = Analyser(datafile, modelfile, usePR=False)
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
            print "%s\tneg %d" % (method,nneg)
            print mgp
            mean_products[method] = {"nneg":nneg, "dim" :dim, "gp": np.array(gp, dtype=np.float32)}
            #mean_products.setdefault(method, {}).setdefault(nneg, {})[dim] = np.array(gp, dtype=np.float32)
        outputfile = os.path.join(args.opdir, args.geometry, "%s.%s"%(args.type, args.dataname))
        with open(outputfile+".p", "wb") as fout:
            pickle.dump({"mean_products":mean_products, "methods":methods[args.dataname], "legendLabels":legendLabels}, fout)
    else:
        outputfile = os.path.join(args.opdir, args.geometry, "%s.%s"%(args.type, args.dataname))
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
        mean_products = result['mean_products']
        legendLabels = result['legendLabels']
        x0 = legendLabels[-1].split("-")[0]
        legendLabels[-1] = "above %s" % x0
        for method, vals in mean_products.iteritems():
            if args.combined:
                outputfile = os.path.join(args.opdir, "combined", args.geometry,  "%s.%s.%s.n%d.d%d"%(args.type, args.dataname, method, vals['nneg'], vals['dim']))
            else:
                outputfile = os.path.join(args.opdir, args.geometry, "%s.%s.%s.n%d.d%d"%(args.type, args.dataname, method, vals['nneg'], vals['dim']))
            if args.geometry in ['length']:
                xlabel = 'length'
            else:
                xlabel = 'atm'
            plotDistribution(vals['gp'], xlabel=xlabel, ylabel="Density", legends=legendLabels, modelName=method, outfile=outputfile, show=False, combined=args.combined)

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    typeAnalysis(args)

if __name__ == "__main__":
	main()
