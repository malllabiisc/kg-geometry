import sys
import os

import argparse
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import scipy.stats as scistats

name_map = {'hole':"HolE", "complex":"ComplEx", "distmult":"DistMult", "transe":"TransE", "transr":"TransR", 'stranse':"STransE", "add":"Additive", "mult":"Multiplicative"}
name_defs = {"atm":"Alignment to Mean (ATM)", "length":"Avg Vector Length", "conicity":"Conicity", "Average conicity":"Average Conicity", "Average length":"Average Vector length"}
#name_defs = {"atm":"Alignment to Mean (ATM)", "length":"Avg Vector Length", "conicity":"Aperture", "Average conicity":"Average Aperture", "Average length":"Average Vector length"}
methods = ['transe','transr','stranse','distmult','hole','complex']
datasets = ["fb15k", "wn18"]
method_types = ["add", "mult"]
data_mtypes = [dmtypes for dmtypes in product(datasets, method_types)]

name_map.update(dict(zip(data_mtypes, ["%s-%s"%(x,y) for x,y in data_mtypes ])))

cone_perf_offsets = {"complex":{1: (-0.02, 0.9), 50: (0.01, 0.1)},
                     "hole"   :{1: (0.01, 0.1), 50: (-0.18,-0.9), 100:(-0.20,-0.1)},
                     "distmult":{1:(-0.02, -2.5), 50:(0.01, -1.1)},
                     "transe" :{1:(0.01,0.1)},
                     "transr": {1:(0.01,-1.1), 50:(0.01, -0.9)},
                     "stranse":{},
                     }
len_perf_offsets = { "complex" :{1:(-1.8, 1.0),  50: (0.1, -0.5), 100:(0.1, -0.6)},
                     "hole"    :{1:(-0.5, 1.0),   50: (0.15,-1), 100:(0.15, 0.0)},
                     "distmult":{1:(-2.4, -0.5), 50:(0.1, -0.5), 100:(0.1, -0.5)},
                     "transe"  :{1:(0.1, -0.5)},
                     "transr"  :{1:(0.1,-1.0),  50:(0.1, -1.2), 100:(0.1, -0.5)},
                     "stranse" :{1:(0.1, -0.5)},
                     }
perf_offsets = {"conicity":cone_perf_offsets, "length":len_perf_offsets}
default_offset = (0.01, -0.5)
borderwidth = 5

colors = "rgbcmykw" #plt.cm.get_cmap("hsv", N)
colors_map = dict(zip(methods,colors[:len(methods)]))
colors_map.update(dict(zip(data_mtypes, colors[:len(data_mtypes)])))
colors_map.update(dict(zip(method_types, colors[:len(method_types)])))

#safe_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#safe_colors = ['#c7c7c7', '#737373', '#bdbdbd', '#525252', '#969696', '#252525', '#f0f0f0', '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00']
#safe_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
#safe_colors = ["#302c21", "#b1beac", "#324340", "#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]
#safe_colors = ["#456a74", "#002a33", "#2178a3", "#0d353f", "#417792", "#274c56", "#2c647e", "#13526c"]
#safe_colors = ["#89a2ec", "#2580fe", "#2b5a9b", "#4372de", "#4795e0", "#3e6cba", "#4f8eed", "#6e88d0"]
#safe_colors = ['#c7c787', "#b1beac", "#324340", "#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]
safe_colors = ["#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#a7bdb7","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2","#d8d0c2", "#696356", "#a7bdb7", "#ab928a", "#8eacb4"]
safe_colors_map = dict(zip(methods,safe_colors[:len(methods)]))
safe_colors_map.update(dict(zip(data_mtypes, safe_colors[:len(data_mtypes)])))
safe_colors_map.update(dict(zip(method_types, safe_colors[:len(method_types)])))

markers = "+.x3ov^<>p"
markers_map = dict(zip(methods,markers[:len(methods)]))
markers_map.update(dict(zip(data_mtypes, markers[:len(data_mtypes)])))
markers_map.update(dict(zip(method_types, markers[:len(method_types)])))

markers_filled = "DsdH*8Px><^Vph"
markers_filled_map = dict(zip(methods,markers_filled[:len(methods)]))
markers_filled_map.update(dict(zip(data_mtypes, markers_filled[:len(data_mtypes)])))
markers_filled_map.update(dict(zip(method_types, markers_filled[:len(method_types)])))

#hatch_patterns = ( '.', '*', 'o','-', '+', 'x', 'O', '\\')
hatch_patterns = ('-', '+', '|', '\\', 'x', '/', 'o', '.', '*', 'O')
hatch_patterns_map = dict(zip(methods,hatch_patterns[:len(methods)]))
hatch_patterns_map.update(dict(zip(data_mtypes, ('-', '|', '\\', '/'))))
hatch_patterns_map.update(dict(zip(method_types, hatch_patterns[:len(method_types)])))
#hatch_patterns_map.update(dict(zip(data_mtypes, hatch_patterns[:len(data_mtypes)])))

boxsizes = [(0.001, 0.945, .998, .1), (0.0, 1.005, 1.0, 0.1)]

def getPerfFromRanks(ranks):
    if len(ranks) < 1:
        return {}
    hits_1 = 100.0*sum(ranks==1)/ranks.shape[0]
    hits_3 = 100.0*sum(ranks<=3)/ranks.shape[0]
    hits_10 = 100.0*sum(ranks<=10)/ranks.shape[0]
    hits_100 = 100.0*sum(ranks<=100)/ranks.shape[0]
    mrr = np.mean([100.0/xx for xx in ranks], axis=0)
    mr = ranks.mean(axis=0)
    perf = {"Hits@1" : hits_1, "Hits@3" : hits_3, "Hits@10": hits_10, "Hits@100" : hits_100, "MR":mr, "MRR" : mrr}
    return perf


class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

def plotConePerf(methods, nnegs, dims, result, outfile=None, show=False, xlabel = "conicity", whitelist=[]):
    key = "Hits@10"
    #key = "MR"
    markermap = dict(zip(methods, markers_filled[:len(methods)]))
    colormap = dict(zip(methods, ["#0f0f0f"]*len(methods))) #dict(zip(methods, safe_colors[:len(methods)]))
    sizemap = dict(zip(dims, [100,200]))
    fontsize = 40
    model_name_size = fontsize*0.67
    linewidth = 10
    fig, ax = plt.subplots()
    #for dim, nneg, method in product(dims, nnegs, methods):
    rects = []
    for method in methods:
        m = markermap[method]
        c = colormap[method]
        xs = []
        ys = []
        names = []
        sizes = []
        addRects = True
        for dim in dims:
            for nneg in nnegs:
                name = "%s.n%d.d%d" % (method, nneg, dim)
                if whitelist and name not in whitelist:
                    continue
                #name = "%d" % (nneg)
                name = "%s(N=%d)" % (name_map[method],nneg)
                perf = result['perfs'][method][dim].get(nneg)
                if perf is None:
                    print "Perf not found for %s" % (name)
                    continue
                conicity = result['mean_products'][dim][nneg][method]
                xs.append(conicity[-1])
                ys.append(perf[key])
                #print "%s.n%d.n%d, %f , %f" % (method, nneg, dim, conicity[-1], perf[key])
                names.append(name)
                sizes.append(sizemap[dim])
                sp = ax.scatter(xs[-1], ys[-1], s=sizes[-1], c=c, marker=m, linewidths=1, cmap="BuPu")
                cur_offset = perf_offsets[xlabel].get(method).get(nneg, default_offset)
                ax.annotate(name, (xs[-1]+cur_offset[0], ys[-1]+cur_offset[1]), fontsize=model_name_size)
                """
                if nneg == 1:
                    if method in ['complex']:
                        ax.annotate(name, (xs[-1]-0.02, ys[-1]+0.9), fontsize=model_name_size)
                    elif method in ['distmult']:
                        ax.annotate(name, (xs[-1]-0.02, ys[-1]-2.5), fontsize=model_name_size)
                    elif method in ['transr']:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]-1.1), fontsize=model_name_size)
                    elif method in ['transe']:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]+0.1), fontsize=model_name_size)
                    elif method in ['hole']:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]+0.1), fontsize=model_name_size)
                    else:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]-0.5), fontsize=model_name_size)
                elif method in ['hole'] and nneg in [50]:
                    ax.annotate(name, (xs[-1]-0.19, ys[-1]-0.9), fontsize=model_name_size)
                elif method in ['hole'] and nneg in [100]:
                    ax.annotate(name, (xs[-1]-0.21, ys[-1]-0.1), fontsize=model_name_size)
                elif nneg == 50:
                    if method in ['complex']:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]+0.1), fontsize=model_name_size)
                    elif method in ['distmult']:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]-1.1), fontsize=model_name_size)
                    elif method in ['transr']:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]-0.9), fontsize=model_name_size)
                    else:
                        ax.annotate(name, (xs[-1]+0.01, ys[-1]-0.5), fontsize=model_name_size)
                else:
                    ax.annotate(name, (xs[-1]+0.01, ys[-1]-0.5), fontsize=model_name_size)
                """
                if addRects:
                    rects.append(sp)
                    addRects = False
        #rects.append(ax.scatter(xs, ys, s=sizes, c=c, marker=m, linewidths=1))
        """
        for i, txt in enumerate(names):
            if xs[i] > 0.8:
                ax.annotate(txt, (xs[i]-0.1, ys[i]-0.5), fontsize=fontsize*0.5)
            else:
                ax.annotate(txt, (xs[i]+0.01, ys[i]-0.5), fontsize=fontsize*0.5)
        """

    ylabel = "Performance(Hits@10)"
    xlabel = name_defs[xlabel]
    legends = [name_map[method] for method in methods]
    ax.set_ylabel(ylabel, fontsize=fontsize, weight='bold')
    ax.set_xlabel(xlabel, fontsize=fontsize, weight='bold')
    plt.xticks(fontsize=fontsize/2, weight='bold')
    plt.yticks(fontsize=fontsize/2, weight='bold')
    for i in ax.spines.itervalues():
        i.set_linewidth(borderwidth*0.7)
    #if legends:
        #ax.legend(rects, legends, loc="upper right", fontsize=fontsize, prop={'weight':'bold', 'size':fontsize*0.67})
        #ax.legend(rects, legends, bbox_to_anchor=(0.001, 0.945, .998, .1), loc=3, ncol=len(methods), mode='expand', borderaxespad=0., fontsize=fontsize)
 
    fig = plt.gcf()
    fig.set_size_inches(16,10)
    if outfile is not None:
        if not outfile.endswith(".png"):
            outfile = outfile+".png"
        plt.savefig(outfile, dpi=100)
    if show:
        plt.show()

def plotDistribution(gp, xlabel="atm", ylabel="Density", legends=None, modelName="", outfile=None, show=True, combined=True):
    markers = "+.x3ov^<>p"
    #markers = markers_filled
    fontsize = 60
    linewidth = 10
    fig, ax = plt.subplots()
    y_interval = 1.0
    if "trans" in modelName.lower():
        #maxy = 5.0 #entities
        maxy = 8.0
    else:
        #maxy = 16.0 #entities
        maxy = 40.0 #relations
        y_interval = 5.0
    #plt.ylim(0, maxy)
    #plt.yticks(np.arange(0, maxy, y_interval), fontsize=fontsize/2)
    figs = []
    #for i, gpi in enumerate(gp):
    npoints = 5
    npointsx = 4
    miny = 0
    maxy = 0
    minx = 0
    maxx = 0
    m,n = gp.shape
    conicity_x = 0
    conicity_y = 0
    conx_offset = 0
    cony_offset = 0
    if modelName in ['transe']:
        maxy = 2
        npoints = 2
        #npointsx = 3
        cony_offset = 0.3
        conx_offset = 0.1
    elif modelName in ['transr']:
        maxy = 2
        npoints = 2
        #npointsx = 3
        cony_offset = 0.3
        conx_offset = 0.0
    elif modelName in ['stranse']:
        maxy = 2
        npoints = 2
        #npointsx = 3
	cony_offset = 0.2
        conx_offset = -0.57
    elif modelName in ['hole']:
        maxy = 2
        npoints = 2
        #maxx = 18
        #npointsx = 6
	conx_offset = -0.1#-0.3
        conx_offset = -0.01
        cony_offset = 0.2
    elif modelName in ['distmult']:
        maxy = 4
        npoints = 4
        #maxx = 12.0
        #npointsx = 6
	conx_offset = -0.2 #0.0
    elif modelName in ['complex']:
        maxy = 12
        npoints = 4
        #maxx = 15.0
        #npointsx=5
	conx_offset = -0.2 #0.0
    if combined:
	gp = gp.reshape(1, m*n)
    for i in np.arange(gp.shape[0]):
        gpi = gp[i,:]
        nSamples = gpi.shape[0]
        density = scistats.gaussian_kde(gpi)
        x,y = np.histogram(gpi, nSamples)
	maxy = max(np.ceil(density(y).max()), maxy)
	maxx = max(np.ceil(y.max()), maxx)
	minx = min(np.floor(y.min()), minx)
        figs.append(ax.plot(y, density(y), c="k", label=legends[i], marker=markers[i], markevery=2, markeredgewidth=10, markersize=30, linewidth=linewidth))
        #figs.append(ax.plot(y, density(y), c="k", label=legends[i], marker=markers[i], markevery=gpi.shape[0]/10, markeredgewidth=10, markersize=30, linewidth=linewidth))
        #figs.append(ax.plot(y, density(y), c=colors[i], label=legends[i], marker=markers[i], markevery=10, markeredgewidth=10, markersize=30, linewidth=linewidth))
	plt.axvline(x=gpi.mean(), ymax=1, color='k', linestyle='--', linewidth=linewidth*0.7)
	#plt.axvline(x=gpi.mean(), ymax=density(gpi.mean())/maxy, color='k', linestyle='--', linewidth=linewidth*0.7)
	conicity_x = gpi.mean()
	conicity_y = density(gpi.mean())

	if conicity_x > 0.2:
	    conicity_x = -conicity_x
	#ax.annotate("Avg Length = %0.2f" %(gpi.mean()), (conicity_x+conx_offset, conicity_y+cony_offset), fontsize=fontsize*0.9, weight='bold')
	ax.annotate("%s = %0.2f" %(name_defs['conicity'], gpi.mean()), (conicity_x+conx_offset, conicity_y+cony_offset), fontsize=fontsize*0.9, weight='bold')

    #plt.legend(figs,  legendLabels, loc='upper right')
    #ax.legend(bbox_to_anchor=(0.001, 1.06, .998, .1), loc=3, ncol=gp.shape[0], mode='expand', borderaxespad=0., fontsize=fontsize*0.55)
    #ax.legend(loc='upper left')
    #plt.legend(figs, legendLabels, loc='upper right')
    for i in ax.spines.itervalues():
        i.set_linewidth(borderwidth)
    if xlabel in ['conicity', 'atm']:
	plt.xlim(-1.0,1.0)
        minx = -1.0
        maxx = 1.0
    else:
	plt.xlim(minx, maxx+1)
        maxx = maxx+1
    xlabel = name_defs[xlabel]
    ax.set_title(name_map[modelName], fontsize=fontsize*1.3, weight='bold', fontdict={"verticalalignment":"bottom"})
    if modelName in ['transe', 'distmult']:
	ax.set_ylabel(ylabel, fontsize=fontsize, weight='bold')
    #if modelName in ['distmult', 'hole', 'complex']:
    ax.set_xlabel(xlabel, fontsize=fontsize, weight='bold')
    """
    if modelName in ['stranse', "transe", "transr"]:
        maxy = 4.0
        npoints = 4
    if modelName in ['distmult', "complex", "hole"]:
        maxy = 8
        npoints = 4
    """
    plt.xticks(np.arange(minx, maxx+0.1, (maxx-minx)/npointsx), fontsize=fontsize*0.9, weight='bold')
    #plt.xticks(list(plt.xticks()[0]) + [gp.mean()])
    plt.yticks(np.arange(0,maxy+0.1, maxy/npoints)[1:], fontsize=fontsize*0.9, weight='bold')
    plt.ylim(miny,maxy)
    fig = plt.gcf()
    fig.set_size_inches(20, 16)
    plt.savefig(outfile+".png", dpi=100)
    if show:
        plt.show()


def plotBars(vals, xlabel, ylabel, legends=[], xticks=[], outfile=None, show=True):
    vals = np.array(vals, dtype=np.float32)
    m, n = vals.shape
    indices = np.arange(m)
    width=0.15
    rects = []
    fontsize=40
    linewidth = 10
    fig, ax = plt.subplots()
    for i in range(n):
        method = legends[i]
        rects.append(ax.bar(indices+width*(i+1), vals[:,i], width=width, color=safe_colors_map[method], hatch=hatch_patterns_map[method]))
        #plt.scatter(indices+width*(i+1.5), vals[:,i]+0.01, marker=markers[i])
    ax.plot([indices[0], indices[-1]+width*(n+2)], [0, 0], "k-")
    ylabel = name_defs[ylabel]
    ax.set_ylabel(ylabel, fontsize=fontsize, weight='bold')
    ax.set_xlabel(xlabel, fontsize=fontsize, weight='bold')
    plt.xticks(fontsize=fontsize*0.8, weight='bold')
    plt.yticks(fontsize=fontsize*0.8, weight='bold')
    for i in ax.spines.itervalues():
        i.set_linewidth(borderwidth)
    if xticks:
        ax.set_xticks(indices+width*((n+3)/2))
        ax.set_xticklabels(xticks)
    if legends:
	"""
        #leg = plt.legend(rects, [name_map[method] for method in legends], bbox_to_anchor=boxsizes[1], loc=3, ncol=n/2, mode='expand', borderaxespad=0., fontsize=fontsize*2/3, prop={'weight':'bold', 'size':fontsize*0.7})
	legLabels = []
	for iidx, method in enumerate(legends):
		legLabels.append(mpatches.Patch(label=name_map[method], hatch=hatch_patterns_map[method], color=safe_colors_map[method], linewidth=1))
	"""
        leg = ax.legend(rects, [name_map[method] for method in legends], bbox_to_anchor=boxsizes[1], loc=3, ncol=3, mode='expand', borderaxespad=0., fontsize=fontsize, prop={'weight':'bold', 'size':fontsize*0.7}, markerscale=10.0)
        #leg = ax.legend(rects, [name_map[method] for method in legends], bbox_to_anchor=boxsizes[1], loc=3, ncol=n/2, mode='expand', borderaxespad=0., fontsize=fontsize, prop={'weight':'bold', 'size':fontsize*0.7}, markerscale=10.0)
    """
    signs = []
    for i in range(m):
        for j in range(n):
            signs.append(np.sign(vals[i,j]))
    def autolabel(rects, iter):
        #Attach a text label above each bar displaying its height
        for rect in rects:
            height = rect.get_height()
            print height
            if signs[iter] > 0:
                ax.text(rect.get_x() + rect.get_width()/2., 1.01*height*signs[iter], '%.3f' % height, ha='center', va='bottom')
            else:
                ax.text(rect.get_x() + rect.get_width()/2., 1.5*height*signs[iter], '%.3f' % -height, ha='center', va='bottom')
    iter = 0
    for rect in rects:
        autolabel(rect, iter)
        iter +=1
    """

    fig = plt.gcf()
    fig.set_size_inches(16,10)
    if outfile is not None:
        if not outfile.endswith(".png"):
            outfile = outfile+".png"
        plt.savefig(outfile, dpi=100)
    if show:
        plt.show()



def normalize(x, axis=1, order=2):
    y = np.linalg.norm(x, axis=axis, ord=order)
    if axis==1:
        return x/y.reshape(x.shape[0],1)
    else:
        return x/y.reshape(1, x.shape[1])

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments")
    parser.add_argument("-a1", "--arg1", type=int, help="argument 1", required=True)
    return parser

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
	main()

