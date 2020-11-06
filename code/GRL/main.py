import pprint
from util import dataloader, data_utils as data_utils
from run_networks import model
import warnings
from util.utils import source_import
from util.knowledge_graph import KnowledgeGraph
from args import args
import os

data_root = {'FB15K-237': '../data/FB15K-237',
             'UMLS':'../data/UMLS'}

def process_data(dataset,add_reverse_relations):
    data_dir = data_root[dataset]
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = os.path.join(data_dir, 'train.triples')
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, add_reverse_relations)

def run_experiment(args):

    args.config = './config/config.py'
    config = source_import(args.config).config
    training_opt = config['training_opt']

    training_opt['dataset'] = args.data
    training_opt['log_dir'] = './logs/'+args.data

    dataset = training_opt['dataset']
    training_opt['batch_size']=args.batch_size
    training_opt['num_epochs']=args.num_epochs
    if not os.path.isdir(training_opt['log_dir']):
        os.makedirs(training_opt['log_dir'])
    if args.preprocess:
        process_data(dataset,training_opt['add_inverse_relation'])
        return 0

    print('Preparing knowledge graph base!')
    kg = KnowledgeGraph(data_root[dataset],args,config)
    kg.load_all_triples(data_root[dataset], add_reversed_edges=training_opt['add_inverse_relation'])
    pprint.pprint(config)

    if args.train:
        print('Loading dataset from: %s' % data_root[dataset])
        data = {x: dataloader.load_data_kg(data_root=data_root[dataset], dataset=dataset, phase=x,
                                           batch_size=training_opt['batch_size'],kg_base=kg,
                                           add_reversed_edges=training_opt['add_inverse_relation'])
                for x in (['train', 'dev'])}
        config['training_opt']['num_classes'] = len(data['train'].dataset.relation2id.keys())

        training_model = model(config, data, test=False, args=args)
        training_model.train()

    elif args.test:
        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        print('Under testing phase, we load training data simply to calculate training data number for each class.')
        data = {x: dataloader.load_data_kg(data_root=data_root[dataset], dataset=dataset, phase=x,
                                        batch_size=training_opt['batch_size'], kg_base=kg,
                                        test_open=False,
                                        shuffle=False,
                                        add_reversed_edges=False)
                for x in ['train', 'dev']}

        config['training_opt']['num_classes'] = len(data['train'].dataset.relation2id.keys())
        training_model = model(config, data, test=True,args=args)
        training_model.load_model()
        training_model.eval_batch(phase='dev', openset=False)

    elif args.test_open:
        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        print('Under testing phase, we load training data simply to calculate training data number for each class.')

        data = {x: dataloader.load_data_kg(data_root=data_root[dataset], dataset=dataset, phase=x,
                                        batch_size=training_opt['batch_size'], kg_base=kg,
                                        test_open=True,
                                        shuffle=False,
                                        add_reversed_edges=False)
                for x in ['train','dev']}
        training_model = model(config, data, test=True,args=args,kg=kg)
        training_model.load_model()
        training_model.eval_batch(phase='dev', openset=args.test_open)

    print('ALL COMPLETED.')

if __name__ == '__main__':

    run_experiment(args)
