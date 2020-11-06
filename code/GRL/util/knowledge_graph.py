import os
import json

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index


class KnowledgeGraph():

    def __init__(self, data_dir, args, config):
        super(KnowledgeGraph, self).__init__()
        self.entity2id, self.id2entity = {}, {}
        self.relation2id, self.id2relation = {}, {}
        self.all_triples = []
        self.train_triples = []
        self.dev_triples = []
        self.test_triples = []
        self.data_dir = data_dir

        self.embmodel = args.embmodel
        self.entity_dim = config['networks']['emb_model']['params']['entity_dim']
        self.relation_dim = config['networks']['emb_model']['params']['relation_dim']
        self.emb_dropout_rate = config['networks']['emb_model']['params']['emb_dropout_rate']

        self.ini_dics()

        self.rel2candidates = {}
        self.rel2e2_notconsider = {}
        if "ONE" in self.data_dir:
            self.rel2candidates_str = json.load(open(self.data_dir + "/rel2candidates.json"))
            for rel in self.rel2candidates_str.keys():
                if rel not in self.relation2id.keys():
                    continue
                rel2candidate = self.rel2candidates_str[rel]
                ents = []
                for ent in rel2candidate:
                    ents.append(self.entity2id[ent])
                self.rel2candidates[self.relation2id[rel]] = ents
                self.rel2e2_notconsider[self.relation2id[rel]] = list(set(self.entity2id.values())-set(ents))
        print("len(self.rel2e2_notconsider)",len(self.rel2e2_notconsider))

    def ini_dics(self):
        self.entity2id, self.id2entity = load_index(os.path.join(self.data_dir, 'entity2id.txt'))
        print('Sanity check: {} entities loaded'.format(len(self.entity2id)))
        self.relation2id, self.id2relation = load_index(os.path.join(self.data_dir, 'relation2id.txt'))
        print('Sanity check: {} relations loaded'.format(len(self.relation2id)))

    def load_all_triples(self, data_dir, add_reversed_edges=False):
        def add_subject(e1, e2, r, d):
            if not e2 in d:
                d[e2] = {}
            if not r in d[e2]:
                d[e2][r] = set()
            d[e2][r].add(e1)

        def add_object(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not r in d[e1]:
                d[e1][r] = set()
            d[e1][r].add(e2)

        def add_relation(e1, e2, r, d):
            if not e1 in d:
                d[e1] = {}
            if not e2 in d[e1]:
                d[e1][e2] = set()
            d[e1][e2].add(r)


        train_subjects, train_objects, train_relations = {}, {}, {}
        dev_subjects, dev_objects, dev_relations = {}, {}, {}
        all_subjects, all_objects, all_relations = {}, {}, {}
        only_dev_objects = {}
        all_triples = {}
        for file_name in ['train.triples', 'dev.triples', 'test.triples']:
            print(file_name)
            with open(os.path.join(data_dir, file_name)) as f:
                for line in f:
                    ttt = line.strip().split()
                    e1 = ttt[0]
                    e2 = ttt[1]
                    r = ttt[2]

                    e1, e2, r = self.triple2ids((e1, e2, r))
                    triple_str = str(e1) + '_' + str(e2) + '_' + str(r)

                    if not triple_str in all_triples:
                        all_triples[triple_str] = 1

                    if file_name in ['train.triples']:
                        add_subject(e1, e2, r, train_subjects)
                        add_object(e1, e2, r, train_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), train_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), train_objects)

                    if file_name in ['train.triples', 'dev.triples']:
                        add_subject(e1, e2, r, dev_subjects)
                        add_object(e1, e2, r, dev_objects)
                        if add_reversed_edges:
                            add_subject(e2, e1, self.get_inv_relation_id(r), dev_subjects)
                            add_object(e2, e1, self.get_inv_relation_id(r), dev_objects)
                    if file_name in ['dev.triples']:

                        add_object(e1, e2, r, only_dev_objects)
                        if add_reversed_edges:
                            add_object(e2, e1, self.get_inv_relation_id(r), only_dev_objects)
                    add_subject(e1, e2, r, all_subjects)
                    add_object(e1, e2, r, all_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, self.get_inv_relation_id(r), all_subjects)
                        add_object(e2, e1, self.get_inv_relation_id(r), all_objects)
        self.train_subjects = train_subjects
        self.train_objects = train_objects
        self.dev_subjects = dev_subjects
        self.dev_objects = dev_objects
        self.all_subjects = all_subjects
        self.all_objects = all_objects
        self.only_dev_objects = only_dev_objects
        self.all_triples = all_triples

    def get_inv_relation_id(self, r_id):
        return self.relation2id[self.id2relation[r_id] + '_inv']

    def id2triples(self, triple):
        e1, e2, r = triple
        return self.id2entity[e1], self.id2entity[e2], self.id2relation[r]

    def triple2ids(self, triple):
        e1, e2, r = triple
        return self.entity2id[e1], self.entity2id[e2], self.relation2id[r]

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)
