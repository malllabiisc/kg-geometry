import sys
import os       
import numpy as np
import cPickle as pickle
import argparse

class Model:
    def __init__(self):
        self.E = None
        self.R = None
        self.eNames = None
        self.rNames = None
        self.modelName = None
        self.other_params = {}

    def setValues(self, E, R, enames, rnames, mname):
        self.E = E
        self.R = R
        self.eNames = enames
        self.rNames = rnames
        self.modelName = mname

    def saveModel(self, filename):
        outdict = {"model":self.modelName, "E":self.E, "R":self.R, "eNames":self.eNames, "rNames":self.rNames}
        outdict.update(self.other_params)
        pickle.dump(outdict, open(filename, 'wb'))

    def loadModel(self, filename):
        x = pickle.load(open(filename, 'rb'))
        if type(x['model']) == str:
            self.E = x['E']
            self.R = x['R']
            self.eNames = x['eNames']
            self.rNames = x['rNames']
            self.modelName = x['model']
            self.fpos_test = x.get('fpos test', [])
            #self.fpos_test = x['fpos test']
            x = None
        else:
            self.E = x['model'].E
            self.R = x['model'].R
            self.fpos_test = x['fpos test']
            self.modelName = "hole"
            self.eNames = [str(i) for i in range(self.E.shape[0])]
            self.rNames = [str(i) for i in range(self.R.shape[0])]
            self.model = x


def loadHolEModel(modelFile, dataFile, outputFile, mname):
    holEModel = pickle.load(open(modelFile, 'rb'))['model']
    model = Model()
    data = pickle.load(open(dataFile, 'rb'))
    model.setValues(holEModel.E, holEModel.R, data['entities'], data['relations'], mname)
    model.saveModel(outputFile)

def loadComplexModel(modelFile, dataFile, outputFile, mname):
    complexModel = pickle.load(open(modelFile, 'rb'))
    model = Model()
    data = pickle.load(open(dataFile, 'rb'))
    model.setValues(complexModel['E'], complexModel['R'], data['entities'], data['relations'], mname)
    model.saveModel(outputFile)

def loadTransEModel(entFile, relFile,  dataFile, outputFile, mname):
    ent2vec = np.array(np.genfromtxt(entFile), dtype=np.float32)
    rel2vec = np.array(np.genfromtxt(relFile), dtype=np.float32)
    model = Model()
    data = pickle.load(open(dataFile, 'rb'))
    model.setValues(ent2vec, rel2vec, data['entities'], data['relations'], mname)
    model.saveModel(outputFile)

def loadSTransEModel(entFile, relFile, mhFile, mtFile, dataFile, outputFile, mname):
    ent2vec = np.array(np.genfromtxt(entFile), dtype=np.float32)
    rel2vec = np.array(np.genfromtxt(relFile), dtype=np.float32)
    ne, de = ent2vec.shape
    nr, dr = rel2vec.shape
    mat_h = np.reshape(np.array(np.genfromtxt(mhFile), dtype=np.float32), [nr, dr, de])
    mat_t = np.reshape(np.array(np.genfromtxt(mtFile), dtype=np.float32), [nr, dr, de])
    model = Model()
    data = pickle.load(open(dataFile, 'rb'))
    model.setValues(ent2vec, rel2vec, data['entities'], data['relations'], mname)
    model.other_params = {"MH":mat_h, "MT":mat_t}
    model.saveModel(outputFile)

def loadTransRModel(entFile, relFile, mFile, dataFile, outputFile, mname):
    ent2vec = np.array(np.genfromtxt(entFile), dtype=np.float32)
    rel2vec = np.array(np.genfromtxt(relFile), dtype=np.float32)
    ne, de = ent2vec.shape
    nr, dr = rel2vec.shape
    mat = np.reshape(np.array(np.genfromtxt(mFile), dtype=np.float32), [nr, dr, de])
    model = Model()
    data = pickle.load(open(dataFile, 'rb'))
    model.setValues(ent2vec, rel2vec, data['entities'], data['relations'], mname)
    model.other_params = {"M":mat}
    model.saveModel(outputFile)

def getParser():
    parser = argparse.ArgumentParser(description="parser for arguments")
    parser.add_argument("-m", "--mdir", type=str, help="directory containing model file(s)", required=True)
    parser.add_argument("-o", "--odir", type=str, help="output directory to dump new pickle", required=True)
    parser.add_argument("-n", "--name", type=str, help="model name", required=True)
    parser.add_argument("-d", "--dataname", type=str, help="name of dataset", required=True)
    parser.add_argument("--nnegs", type=int, nargs="+", help="#negatives", default=[1,50,100])
    parser.add_argument("--dims", type=int, nargs="+", help="#dimensions", default=[50,100,200])
    return parser

def main():
    parser = getParser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    basedir = args.mdir
    modelFn = { "hole":loadHolEModel,
            "complex":loadComplexModel, 
            "distmult":loadComplexModel, 
            "transe":loadTransEModel, 
            "stranse":loadSTransEModel,
            "transr":loadTransRModel
            }
    #for nneg in [1, 50, 100]:
        #for dim in [50, 100, 200]:
    for nneg in args.nnegs:
        for dim in args.dims:
            #modelFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "STransE.s%d.neg%d.r0.0001.m1.l1_1.e2000.entity2vec"%(dim, nneg))
            if args.name.lower() in ['transe']:
                entFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "entity2vec.bern")
                relFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "relation2vec.bern")
                modelargs = {"entFile":entFile, "relFile":relFile, "mname":args.name}
                modelFile = entFile
            elif args.name.lower() in ['transr']:
                entFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "entity2vec.bern")
                relFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "relation2vec.bern")
                matFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "A.bern")
                modelargs = {"entFile":entFile, "relFile":relFile, "mname":args.name, "mFile":matFile}
                modelFile = entFile
            elif args.name.lower() in ['stranse']:
                "STransE.s50.r0.0001.m1.0.l1_1.i_0.e2000"
                "STransE.s%d.r0.0005.m5.0.l1_1.e2000"
                #prefix = "STransE.s%d.r0.0001.m1.0.l1_1.i_0.e2000"
                prefix = "STransE.s%d.neg%d.r0.0001.m1.0.l1_1.i_0.e2000"
                #prefix= "STransE.s%d.r0.0005.m5.0.l1_1.e2000"
                #prefix = "STransE.s%d.neg%d.r%s.m%s.l%s.e%s"
                #prefix = "STransE.s%d.r%s.m%s.l%s.i_0.e%s"
                m = "1"
                r = "0.0001"
                l = "1_1"
                e = "2000"
                #name = prefix%(dim, nneg, r, m, l, e)
                name = prefix%(dim, nneg)
                #name = prefix%(dim, r, m, l, e)
                entFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, name+".entity2vec")
                relFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, name+".relation2vec")
                mhFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, name+".W1")
                mtFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, name+".W2")
                modelargs = {"entFile":entFile, "relFile":relFile, "mname":args.name, "mhFile":mhFile, "mtFile":mtFile}
                modelFile = entFile
            elif args.name.lower() in ['hole']:
                modelFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "%s.%s.p"%(args.dataname, args.name))
                modelargs = {"modelFile":modelFile, "mname":args.name}
            elif args.name.lower() in ['complex', 'distmult']:
                "model.l2_0.030000.e_200.lr_0.500000.lc_0.000000.p"
                "model.l2_0.030000.e_100.lr_0.500000.lc_0.000000.p"
                "model.l2_0.010000.e_100.lr_0.100000.lc_0.000000.p"
                "model.l2_0.030000.e_200.lr_0.500000.lc_0.000000.p"
                 
                modelFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "model.l2_0.030000.e_%d.lr_0.500000.lc_0.000000.p"%dim)
                modelargs = {"modelFile":modelFile, "mname":args.name}

            dataFile = "./data/%s.%s.bin" % (args.dataname, args.name)
            outputFile = os.path.join(args.odir, "%s.%s.n%d.d%d.p" % (args.dataname, args.name, nneg, dim))

            if not os.path.exists(modelFile):
                print "file not there"
                print modelFile 
                continue
            if os.path.exists(outputFile):
                print "File already exists"
                print outputFile
                continue
            
            modelargs['dataFile'] = dataFile
            modelargs['outputFile'] = outputFile

            #load the model
            print "Model : %s\tData : %s\tDim : %d\tNeg : %d" % (args.name, args.dataname, dim, nneg)
            modelFn[args.name](**modelargs)


    """
    baseDir = "/scratch/home/chandrahas/JointEmbedding.2017/complEx/complex/new_results"
    modelList = {
                 "./DM/neg_1/dim_50/model.l2_0.010000.e_50.lr_0.100000.lc_0.000000.p" : { 'model': "distmult", 'neg':1, 'dim':50},
                 "./DM/neg_1/dim_100/model.l2_0.010000.e_100.lr_0.100000.lc_0.000000.p" : { 'model': "distmult", 'neg':1, 'dim':100},
                 "./DM/neg_1/dim_200/model.l2_0.010000.e_200.lr_0.100000.lc_0.000000.p" : { 'model': "distmult", 'neg':1, 'dim':200},
                 "./ComplEx/neg_1/dim_50/model.l2_0.010000.e_50.lr_0.100000.lc_0.000000.p" : { 'model': "complex", 'neg':1, 'dim':50},
                 "./ComplEx/neg_1/dim_100/model.l2_0.010000.e_100.lr_0.100000.lc_0.000000.p" : { 'model': "complex", 'neg':1, 'dim':100}
                 #"./DM/neg_1/dim_50/model.l2_0.010000.e_200.lr_0.100000.lc_0.000000.p" : { 'model': "DistMult", 'neg':1, 'dim':50},
                 #"./DM/neg_1/dim_50/model.l2_0.010000.e_100.lr_0.100000.lc_0.000000.p" : { 'model': "DistMult", 'neg':1, 'dim':50},
                }
    for filename, mdesc in modelList.iteritems():
        outputFile = "./data/fb15k.%s.n%d.d%d.p" % (mdesc['model'], mdesc['neg'], mdesc['dim'])
        print "Saving model %s to %s ..." %(filename, outputFile)
        loadComplexModel(os.path.join(baseDir, filename), "./data/fb15k.complex.bin", outputFile) 

    baseDir = "/scratch/home/chandrahas/JointEmbedding/TransR/Relation_Extraction"
    modelList = [
                 #"TransR" : {'ent':"TransR/entity2vec.bern", "rel":"TransR/relation2vec.bern", "neg":1, "dim":100},
                 ("TransE", {'ent':"entity2vec.bern", "rel":"relation2vec.bern", "neg":100, "dim":50}),
                 ("TransE", {'ent':"entity2vec.bern", "rel":"relation2vec.bern", "neg":100, "dim":100}),
                 ("TransE", {'ent':"entity2vec.bern", "rel":"relation2vec.bern", "neg":100, "dim":200}),
                 #"TransH" : {'ent':"TransH/entity2vec.txtbern", "rel":"TransH/relation2vec.txtbern", "neg":1, "dim":100},
                ]
    for mname, mdesc in modelList:
        outputfile = "./data/fb15k.%s.n%d.d%d.p" % (mname, mdesc['neg'], mdesc['dim'])
        entfile = os.path.join(baseDir, mname, "results", "neg_%d"%mdesc['neg'], "dim_%d"%mdesc['dim'], mdesc['ent'])
        relfile = os.path.join(baseDir, mname, "results", "neg_%d"%mdesc['neg'], "dim_%d"%mdesc['dim'], mdesc['rel'])
        datafile = "./data/fb15k.TransE.bin"
        print "Saving model %s to %s ..." % (mname, outputfile)
        loadTransModel(entfile, relfile, datafile, outputfile, mname)

    basedir = "/scratch/home/chandrahas/JointEmbedding.2017/STransE/Datasets/FB15k"
    entfile = os.path.join(basedir, "STransE.s100.r0.0001.m1.l1_1.e2000.entity2vec")
    relfile = os.path.join(basedir, "STransE.s100.r0.0001.m1.l1_1.e2000.relation2vec")
    neg = 1
    dim = 100
    mname = "STransE"
    datafile = "./data/fb15k.TransE.bin"
    outputfile = "./data/fb15k.%s.n%d.d%d.p" % (mname, neg, dim)
    loadSTransEModel(entfile, relfile, datafile, outputfile, mname)

    basedir = "/scratch/home/chandrahas/JointEmbedding.2017/HOLE/holographic-embeddings/results_geometry"
    for i in [50, 100, 200]:
        modelFile = os.path.join(basedir, "neg_1", "dim_%d"%i, "fb15k.hole.p")
        dataFile = "./data/fb15k.bin"
        outputFile = "./data/fb15k.hole.n1.d%d.p" % i
    loadHolEModel(modelFile, dataFile, outputFile)
    basedir = "./embeddings/fb15k/TransR"
    for nneg in [1, 50, 100]:
        for dim in [50, 100, 200]:
            #modelFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "STransE.s%d.neg%d.r0.0001.m1.l1_1.e2000.entity2vec"%(dim, nneg))
            entFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "entity2vec.bern")
            relFile = os.path.join(basedir, "neg_%d"%nneg, "dim_%d"%dim, "relation2vec.bern")
            dataFile = "./data/fb15k.TransE.bin"
            outputFile = "./data/fb15k.TransR.n%d.d%d.p" % (nneg, dim)
            if not os.path.exists(entFile):
                print "file not there"
                print entFile 
                continue
            if os.path.exists(outputFile):
                print "File already exists"
                print outputFile
                continue
            print "Setting dim : %d\t neg : %d" % (dim, nneg)
            mname="TransR"
            loadTransRModel(entFile, relFile, dataFile, outputFile, mname)
    """

if __name__ == "__main__":
    main()
