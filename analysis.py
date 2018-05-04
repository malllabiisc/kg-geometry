import sys
import os

import argparse

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
import cPickle as pickle


class Analyser:
    def __init__(self, datafile, modelfile, usePR=False):
        self.datafile = datafile
        self.modelfile = modelfile
        self.t = Triples(datafile)
        self.model = Model()
        self.model.loadModel(modelfile)
        self.stats = Stats(self.t, usePR)
        self.meanE = self.model.E.mean(axis=0)
        self.meanR = self.model.R.mean(axis=0)

    def getEntIdxs(self, ranges):
        idxs = []
        for rankBand, ns in ranges:
            idxs.append(self.stats.getEnts(rankBand, ns))
        return idxs

    def getRelIdxs(self, ranges):
        idxs = []
        for rankBand, ns in ranges:
            idxs.append(self.stats.getRels(rankBand, ns))
        return idxs

    def entPerf(self, opdir):
        #eRanges = [((0,100), nSamples), ((100,500), nSamples), ((500,5000), nSamples), ((5000, -1), nSamples)]
        eRanges = [(0,100), (100,500), (500, 5000), (5000, self.t.ne)]
        entIndices = []
        for rankband in eRanges:
            entIndices.append(self.stats.getEnts(rankband, rankband[1]-rankband[0]))

        rel_triples = self.t.groupByRelation("test")
        ranks = {}
        ent_perf = {}
        for rel, val in self.model.fpos_test.iteritems():
            for idx, (h,t) in enumerate(rel_triples[rel]):
                ranks.setdefault(h, {}).setdefault('head', []).append((val['head'][idx], val['tail'][idx]))
                ranks.setdefault(t, {}).setdefault('tail', []).append((val['head'][idx], val['tail'][idx]))
        all_ranks = []
        for rangeIdx, idxSet in enumerate(entIndices):
            cur_head_ranks = []
            cur_tail_ranks = []
            cur_all_ranks  = []
            for idx in idxSet:
                cur_head_ranks.extend(ranks.get(idx, {}).get('head', []))
                cur_tail_ranks.extend(ranks.get(idx, {}).get('tail', []))
            cur_all_ranks = cur_head_ranks + cur_tail_ranks
            all_ranks.extend(cur_all_ranks)
            ent_perf[eRanges[rangeIdx]] = {"head" : getPerfFromRanks(np.array(cur_head_ranks, dtype=np.int32)),
                                           "tail": getPerfFromRanks(np.array(cur_tail_ranks, dtype=np.int32)),
                                           "all" : getPerfFromRanks(np.array(cur_all_ranks, dtype=np.int32)),
                                        }
        all_perf = getPerfFromRanks(np.array(all_ranks, dtype=np.int32))
        outfile = os.path.join(opdir, ".".join(os.path.split(self.modelfile)[1].split(".")[:-1]+["ent_perf","p"]))
        with open(outfile, "wb") as fout:
            pickle.dump({"ent_perf":ent_perf, "all_perf":all_perf}, fout)
        outfile = os.path.join(opdir, ".".join(os.path.split(self.modelfile)[1].split(".")[:-1]+["ent_perf","txt"]))
        with open(outfile, "w") as fout:
            fout.write("Range\t\tMR\tMRR\tHits@1\tHits@3\tHits@10\tHits@100\n")
            for a in eRanges:
                perf = ent_perf[a]['all']
                line = "%10s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % (str(a), perf['MR'][1], perf['MRR'][1], perf['Hits@1'][1], perf['Hits@3'][1], perf['Hits@10'][1], perf['Hits@100'][1])
                fout.write(line)

    def relPerf(self, opdir):
        rRanges = []
        interval = 4
        for i in range(0,self.t.nr-1,interval):
            rRanges.append(((i,i+interval), interval))
        #rRanges = [((0,50), 50), ((50,100), 50), ((100,200), 100), ((200, 500), 300), ((500,self.t.nr), self.t.nr-500)]
        #rRanges = [((0,100), 100), ((100,500), 400), ((500,self.t.nr), self.t.nr-500)]
        relIndices = self.getRelIdxs(rRanges)
        idxSets = []
        for rankBand, ns in rRanges:
            idxSets.append(self.stats.getRels(rankBand, ns))
        rel_perf = {}
        all_ranks = self.model.fpos_test
        for rangeIdx, idxSet in enumerate(idxSets):
            cur_ranks = []
            for idx in idxSet:
                cur_ranks.extend(all_ranks.get(idx, {}).get('tail', []))
            rel_perf[rRanges[rangeIdx][0]] = getPerfFromRanks(np.array(cur_ranks, dtype=np.int32))
            #rel_perf.append(getPerfFromRanks(np.array(cur_ranks, dtype=np.int32)))
        outfile = os.path.join(opdir, ".".join(os.path.split(self.modelfile)[1].split(".")[:-1]+["rel_perf","p"]))
        #outfile = os.path.join(os.path.split(self.modelfile)[0], "rel_perf.p")
        with open(outfile, "wb") as fout:
            pickle.dump(rel_perf, fout)
        #outfile = os.path.join(os.path.split(self.modelfile)[0], "rel_perf.txt")
        outfile = os.path.join(opdir, ".".join(os.path.split(self.modelfile)[1].split(".")[:-1]+["rel_perf","txt"]))
        with open(outfile, "w") as fout:
            fout.write("Range\t\tMR\tMRR\tHits@1\tHits@3\tHits@10\tHits@100\n")
            for a,b in rRanges:
                perf = rel_perf[a]
                line = "%10s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n" % (str(a), perf['MR'], perf['MRR'], perf['Hits@1'], perf['Hits@3'], perf['Hits@10'], perf['Hits@100'])
                fout.write(line)

    def run(self, vectorType, sampleMean, isnormalized, outputdir, showplot):

        outputfile = ".".join(os.path.split(self.modelfile)[-1].split(".")[:-1])
        #outputfile = outputfile + ".p"
        #outputfile = outputfile + ".png"
        outputfile = os.path.join(outputdir, outputfile)
        if os.path.exists(outputfile):
            print "File already exists. Exitting..."
            print outputfile
            #return 

        #finalize the set to be analysed
        nSamples = 100
        #eRanges = [((0,100), nSamples), ((100,500), nSamples), ((500,5000), nSamples), ((5000, 50000), nSamples), ((50000, -1), nSamples)]
        eRanges = [((0,100), nSamples), ((100,500), nSamples), ((500,5000), nSamples), ((5000, -1), nSamples)]
        entIndices = self.getEntIdxs(eRanges)
        rRanges = [((0,100), nSamples), ((100,500), nSamples), ((500,-1), nSamples)]
        relIndices = self.getRelIdxs(rRanges)

        #colors = ['r','g','b','c']
        #colors = "rgbcmykw" #plt.cm.get_cmap("hsv", N)
        #legendLabels = ["0-100", "100-500", "500-5000", "5000-"]
        legendLabels=[]
        for a,b in eRanges:
            curLabel = "%d-%d"%(a[0],a[1])
            legendLabels.append(curLabel)
        #markers = ["+", ".", "x", 3]
        markers = "+.x3ov^<>p"

        """
        plt.figure(1)
        plt.suptitle(self.model.modelName + " - TSNE")
        if vectorType in ["ent"]:
            self.runTSNE(entIndices, True)
        else:
            self.runTSNE(relIndices, False)

        plt.figure(2)
        plt.suptitle(self.model.modelName + " - PCA")
        if vectorType in ["ent"]:
            self.runPCA(entIndices, True)
        else:
            self.runPCA(relIndices, False)


        """
        if vectorType in ["ent"]:
            gp, lp = self.getInnerProducts(entIndices, sampleMean=sampleMean, normalized=isnormalized)
        else:
            gp, lp = self.getInnerProducts(relIndices, ent=False, normalized=isnormalized)
        nBuckets = len(gp)

        params = os.path.split(self.modelfile)[-1].split(".")[:-1]
        products = " ".join(["%.4f" % lpp for lpp in lp])
        outstr = "%s %d %d %s" % (params[1], int(params[3][1:]), int(params[2][1:]), products)
        print outstr

        plt.figure(3)
        message = ["Dot Product with", "Global Mean"]
        if isnormalized:
            message[0] = "Normalized "+ message[0]
        if sampleMean:
            message[1] = "Sample Mean"

        plt.title(self.model.modelName)
        #plt.title(self.model.modelName + " - %s"%(" ".join(message)), loc='center')
        #plt.suptitle(self.model.modelName + " - Dot Product with Global Mean")
        plt.xlim(-1.0,1.0)
        if "trans" in self.model.modelName.lower():
            #maxy = 5.0 #entities
            maxy = 3.0
        else:
            #maxy = 16.0 #entities
            maxy = 8.0 #relations
        plt.ylim(0, maxy)
        plt.yticks(np.arange(maxy))
        figs = []
        for i, gpi in enumerate(gp):
            #plt.subplot(nBuckets, 1, i+1)
            density = scistats.gaussian_kde(gpi)
            #x,y, _ = plt.hist(gpi, nSamples)
            #plt.plot(y, density(y), c='r')
            x,y = np.histogram(gpi, nSamples/2)
            figs.append(plt.plot(y, density(y), c=colors[i], label=legendLabels[i], marker=markers[i]))
        #plt.legend(figs,  legendLabels, loc='upper right')
        #plt.legend(loc='upper left')
        #plt.legend(figs, legendLabels, loc='upper right')


        """
        plt.figure(4)
        plt.suptitle(self.model.modelName + " - Dot Product with Local Means")
        for i in range(nBuckets):
            for j in range(nBuckets):
                plt.subplot(nBuckets, nBuckets, nBuckets*i + j + 1)
                plt.xlim(-1,1)
                plt.hist(lp[i][j])
        """

        if vectorType in ['rel']:
            outputfile += ".rel"
        fig = plt.gcf()
        fig.set_size_inches(16,10)
        plt.savefig(outputfile+".png", dpi=72)
        pickle.dump({"model":params[1], "dim":int(params[3][1:]), "neg":int(params[2][1:]), "dots":products},open(outputfile+".p", "wb"))
        if showplot:
            print outputfile
            plt.show()

    def runTSNE(self, indices, ent=True):
        if ent:
            vectors = self.model.E
        else:
            vectors = self.model.R
        nComponents = 2
        dim = vectors.shape[1]
        colors = ['r','g','b','c']

        allIndices = []
        for idxs in indices:
            allIndices.extend(idxs)

        #temp = tsne(vectors[allIndices,:], 2, dim, 20.0)
        temp = TSNE(n_components=2).fit_transform(vectors[allIndices,:])
        for iteration, idxs in enumerate(indices):
            nSamples = len(idxs)
            plt.scatter(temp[iteration*nSamples:(iteration+1)*nSamples,0], temp[iteration*nSamples:(iteration+1)*nSamples,1], c=colors[iteration], marker="o")

        #plt.show()
            
        
    def getInnerProducts(self, indices, sampleMean=False, ent=True, normalized=False):
        if ent:
            vectors = self.model.E
            mean = self.meanE
        else:
            vectors = self.model.R
            mean = self.meanR

        localProducts = []
        globalProducts = []
        meanDotProducts = []

        if sampleMean:
            means = [vectors[index, :].mean(axis=0) for index in indices]
            mean = np.mean(means, axis=0)

        if normalized:
            vectors = normalize(vectors)
            mean = mean/np.linalg.norm(mean)

        for index in indices:
            x = np.dot(vectors[index,:], mean)
            globalProducts.append(x)
            meanDotProducts.append(x.mean())

        meanDotProducts.append(np.mean(meanDotProducts))

        """
        for index1 in indices:
            curVectors = vectors[index1,:]
            curMean = curVectors.mean(axis=0)
            curMean = curMean/np.linalg.norm(curMean)
            curProducts = []
            for index2 in indices:
                curProducts.append(np.dot(vectors[index2,:], curMean))
            localProducts.append(curProducts)
        """
        return globalProducts, meanDotProducts
        #return globalProducts, localProducts

    def getLengths(self, indices, ent=True):
        if ent:
            vectors = self.model.E
        else:
            vectors = self.model.R

        vectorLengths = []
        meanVectorLengths = []

        for index in indices:
            x = np.linalg.norm(vectors[index,:], axis=1, ord=2)
            vectorLengths.append(x)
            meanVectorLengths.append(x.mean())

        meanVectorLengths.append(np.mean(meanVectorLengths))

        return vectorLengths, meanVectorLengths

    def runPCA(self, entIndices, ent=True):
        nComponents = 2
        colors = ['r','g','b','c']
        pca = PCA(n_components = nComponents)
        if ent:
            vectors = self.model.E
        else:
            vectors = self.model.R
        for iteration, idxs in enumerate(entIndices):
            nSamples = len(idxs)
            temp = pca.fit_transform(vectors[idxs,:]) 
            plt.scatter(temp[:,0], temp[:,1], c=colors[iteration], marker="v")
            iteration += 1
        #plt.show()
        
        

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datafile", type=str, help="pickled triples file", required=True)
    parser.add_argument("-m", "--modelfile", type=str, help="pickled model file", required=True)
    parser.add_argument("--smean", dest='smean', help="Flag for using sample mean[default]", action='store_true')
    parser.add_argument("--no-smean", dest='smean', help="Flag for using global mean", action='store_false')
    parser.set_defaults(smean=True)
    parser.add_argument("--normalized", dest="normalized", help="Flag for using normalized vectors for dot products[default]", action='store_true')
    parser.add_argument("--no-normalized", dest="normalized", help="Flag for using unnormalized vectors for dot products", action='store_false')
    parser.set_defaults(normalized=True)
    parser.add_argument("--show", dest="show", help="Flag for showing plot[False]", action='store_true')
    parser.set_defaults(show=False)
    parser.add_argument("--pr", dest="pr", help="Flag for using pagerank plot[False]", action='store_true')
    parser.set_defaults(pr=False)
    parser.add_argument("-t", "--type", type=str, help="[ent]/rel", default="ent")
    parser.add_argument("-o", "--opdir", type=str, help="output directory to save the figure", required=True)
    return parser

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.modelfile):
        print "file doesn't exists : ", args.modelfile
        sys.exit(1)
    a = Analyser(args.datafile, args.modelfile, args.pr)
    #a.runPCA()
    a.run(args.type, args.smean, args.normalized, args.opdir, args.show)
    #a.relPerf(args.opdir)
    #a.entPerf(args.opdir)


if __name__ == "__main__":
	main()

