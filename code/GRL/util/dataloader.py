from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch

class LT_Dataset(Dataset):

    def __init__(self, root, txt, kg, phase, add_reversed_edges=False, test_open=False):
        self.img_path = []
        self.triples = []
        self.labels = []
        self.triples_ini = []
        self.labels_ini = []
        self.labels_stage2_train = []
        self.root = root
        self.txt = txt

        self.entity2id = kg.entity2id
        self.relation2id = kg.relation2id
        self.relation_inv2id = {}
        self.rel2candidates=kg.rel2candidates
        self.rel2e2_notconsider=kg.rel2e2_notconsider
        if phase == 'train':
            self.train_subjects = kg.train_subjects
            self.train_objects = kg.train_objects
            self.all_triples = kg.all_triples
            existing_data = {}
            load_number = 0
            with open(txt) as f:
                for line in f:
                    load_number += 1
                    self.triples_ini.append(line.rsplit('\n')[0].split('\t')[:3])
                    head_entity, tail_entity, relation = line.rsplit('\n')[0].split('\t')[:3]
                    E1 = self.entity2id[head_entity]
                    E2 = self.entity2id[tail_entity]
                    R = self.relation2id[relation]
                    label_ini = list(self.train_objects[E1][R])
                    e_r = str(E1) + '_' + str(R)

                    if e_r not in existing_data:
                        existing_data[e_r] = 1
                        self.labels_ini.append(label_ini)
                        self.labels_stage2_train.append(1)
                        self.triples.append([E1, E2, R])

                    if add_reversed_edges:
                        R_inv = self.relation2id[relation + '_inv']
                        label_ini = list(self.train_objects[E2][R_inv])
                        e_r = str(E2) + '_' + str(R_inv)
                        if e_r not in existing_data:
                            existing_data[e_r] = 1
                            self.labels_stage2_train.append(1)
                            self.labels_ini.append(label_ini)
                            self.triples.append([E2, E1, R_inv])
            print('Loading training data ' + str(len(self.triples)))

        elif phase == 'dev':
            self.seen_entities = kg.train_objects
            self.dev_subjects = kg.dev_subjects
            self.dev_objects = kg.dev_objects
            self.all_subjects = kg.all_subjects
            self.all_objects = kg.all_objects
            self.only_dev_objects = kg.only_dev_objects
            with open(txt) as f:
                for line in f:
                    self.triples_ini.append(line.rsplit('\n')[0].split('\t')[:3])
                    head_entity, tail_entity, relation = line.rsplit('\n')[0].split('\t')[:3]
                    E1 = self.entity2id[head_entity]
                    E2 = self.entity2id[tail_entity]
                    if 'NELL' in txt:
                        if E1 not in self.seen_entities or E2 not in self.seen_entities:
                            continue
                    if test_open:
                        R = 0
                        label_ini = []
                    else:
                        R = self.relation2id[relation]
                        label_ini = list(self.dev_objects[E1][R])
                        e_r = str(E1) + '_' + str(R)
                    self.labels_ini.append(label_ini)
                    self.triples.append([E1, E2, R])

            print('Loading dev data ' + str(len(self.triples)))
        elif phase == 'test':
            self.seen_entities = kg.train_objects
            self.all_subjects = kg.all_subjects
            self.all_objects = kg.all_objects
            with open(txt) as f:
                for line in f:
                    self.triples_ini.append(line.rsplit('\n')[0].split('\t')[:3])
                    head_entity, tail_entity, relation = line.rsplit('\n')[0].split('\t')[:3]
                    E1 = self.entity2id[head_entity]
                    E2 = self.entity2id[tail_entity]
                    if test_open:
                        R = 0
                    else:
                        R = self.relation2id[relation]
                    if 'NELL' in txt:
                        if E1 not in self.seen_entities or E2 not in self.seen_entities:
                            continue

                    label_ini = []
                    label = [0] * len(self.entity2id.keys())
                    self.labels_ini.append(label_ini)
                    self.triples.append([E1, E2, R])
                    self.labels.append(label)

            print('Loading testing data ' + str(len(self.triples)))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        sample = torch.Tensor(self.triples[index]).long()
        truth_pos = self.labels_ini[index]
        label = [0] * len(self.entity2id.keys())
        for pos in truth_pos:
            label[int(pos)] = 1
        label = torch.Tensor(label).view(-1).long()
        if len(self.labels_stage2_train) != 0:
            label_stage2_train = self.labels_stage2_train[index]
        else:
            label_stage2_train = 0
        return sample, label, label_stage2_train

def load_data_kg(data_root, dataset, phase, batch_size, kg_base, sampler_dic=None, test_open=False,
                 shuffle=True, add_reversed_edges=True, stage=1):

    if phase=='dev' and test_open:
        txt = data_root+'/open.triples'
        print('Loading data from %s' % (txt))
        set_ = LT_Dataset(data_root, txt, kg_base, phase, test_open=test_open, add_reversed_edges=add_reversed_edges)
    else:
        txt=data_root+'/%s.triples' % (phase)
        print('Loading data from %s' % (txt))
        set_ = LT_Dataset(data_root, txt, kg_base, phase, add_reversed_edges=add_reversed_edges)


    print('Shuffle is %s.' % (shuffle))
    return DataLoader(dataset=set_, batch_size=batch_size,
                      shuffle=shuffle, num_workers=4)

