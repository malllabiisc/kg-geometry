# Towards Understanding Geometry of Knowledge Graph Embeddings
This is the code for generating the results in the paper "Towards Understanding Geometry of Knowledge Graph Embeddings" to be presented at 56th Annual Meeting of the Association for Computational Linguistics at Melbourne, July 15 to July 20, 2018.

## Required data format
The analysis requires pre-trained KG embeddings along with the KG triples data.
The KG triples data should be a pickle (python2.7) file named `"<dataset>.<method>.bin"`. It should contain following key values:
  1.  `'train_subs'`: list of KG triples used for training, in (head_entity_index, tail_entity_index, relation_index) format.
  2.  `'valid_subs'`: list of KG triples used for validation, in (head_entity_index, tail_entity_index, relation_index) format.
  3.  `'test_subs'`: list of KG triples used for testing, in (head_entity_index, tail_entity_index, relation_index) format.
  4.  `'relations'`: list of KG relations.
  5.  `'entities'`: list of KG entities.
  
The KG embeddings should be stored as pickle (python2.7) file named `"<dataset>.<method>.n<no-of-negatives>.d<dimension>.p"`. It should contain following key values:
  1.  `'rNames'` : list of KG relations.
  2.  `'eNames'` : list of KG entities.
  3.  `'E'` : numpy array of size (numEntities X dimension) containing entity embeddings.
  4.  `'R'` : numpy array of size (numRelations X dimension) containing relation embeddings.
  5.  `'model'` : model name.
  6.  `'fpos test'` : ranks of head and tail entities obtained during link prediction. It is required for performance analysis. It should be a dictionary with relation index as keys, e.g. `{rel_id1 :{'head':[head_rank_1, head_rank_2, ...], 'tail':[tail_rank_1, tail_rank_2, ...]}`.
  
  ## Running type analysis
  For running type analysis (Section 5.1 in the paper), please run following command:
  1.  `python typeAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel>`
  2.  `python typeAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel> --result` (for generating the plots)
  
  ## Running negative analysis
  For running negative analysis (Section 5.2 in the paper), please run following command:
  1.  `python negativeAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel>`
  2.  `python negativeAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel> --result` (for generating the plots)
  
  ## Running dimension analysis
  For running dimension analysis (Section 5.3 in the paper), please run following command:
  1.  `python dimensionAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel>`
  2.  `python dimensionAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel> --result` (for generating the plots)
  
  ## Running performance analysis
  For running performance analysis (Section 5.4 in the paper), please run following command:
  1.  `python perfAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel> -p <performance-file>`
  2.  `python perfAnalysis.py -m <data-directory> -d <dataset-name> -g <conicity/length> --opdir <output-directory> --type <ent/rel> --result -p <performance-file>` (for generating the plots)
  Here the <performance-file> is a pickled file containing performance of different models. It is a nested dictionary and `perf['<method>'][<dimension>][<numNegatives>]` should contain performance `{'MRR':<MRR-value>, 'MR':<MR-value>, 'Hits@10':<Hits@10-value>}` for `<method>` with vector size `<dimension>` and `<numNegatives>` number of negative samples.
  
