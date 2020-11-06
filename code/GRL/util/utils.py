import numpy as np
import torch
import importlib
import time
START_RELATION = 'START_RELATION'
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def print_write2(print_str, log_file):
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()

def change_log_dir(training_opt, args):
    datemark = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    datesetmark = str(training_opt['dataset'])
    embmodel = args.embmodel
    batch_sizemark = str(training_opt['batch_size'])
    epochmark = str(training_opt['num_epochs'])
    lr = str(args.lr)
    newname = 'log_' + datemark + '_' + datesetmark + '_' + embmodel + '_' + batch_sizemark + '_' + epochmark + '_' + lr + '.txt'
    return newname

def hits_and_ranks_all(examples, scores, all_answers, threadhold, rel2e2_notconsider=None, openset=False):
    assert (len(examples) == scores.shape[0])
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    if not openset:
        for i, example in enumerate(examples):
            e1, e2, r = torch.Tensor.cpu(example[0]).numpy(), torch.Tensor.cpu(example[1]).numpy(), torch.Tensor.cpu(
                example[2]).numpy()
            e2_notconsider = []
            if len(rel2e2_notconsider) > 0:
                e2_notconsider = rel2e2_notconsider[int(r)]
            e2_list = all_answers[int(e1)][int(r)]
            e2_multi = []
            for e2entity in e2_list:
                e2_multi.append(int(e2entity))

            e2_multi = dummy_mask + list(e2_list) + list(e2_notconsider)
            # save the relevant prediction
            target_score = float(scores[i, e2])
            # mask all false negatives
            scores[i, e2_multi] = 0
            # write back the save prediction
            scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), 1000))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr = 0,0,0,0,0
    hits_at_1_few, hits_at_3_few, hits_at_5_few, hits_at_10_few, mrr_few = 0,0,0,0,0
    hits_at_1_many, hits_at_3_many, hits_at_5_many, hits_at_10_many, mrr_many = 0,0,0,0,0
    count=0
    count_few = 0
    count_many = 0
    for i, example in enumerate(examples):
        e1, e2, r = torch.Tensor.cpu(example[0]).numpy(), torch.Tensor.cpu(example[1]).numpy(), torch.Tensor.cpu(
            example[2]).numpy()
        pos = np.where(top_k_targets[i] == e2)[0]
        if r > threadhold:
            count_few += 1
        else:
            count_many+=1
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += (1.0) / (pos + 1)
            count+=1
            if r > threadhold:
                if pos < 10:
                    hits_at_10_few += 1
                    if pos < 5:
                        hits_at_5_few += 1
                        if pos < 3:
                            hits_at_3_few += 1
                            if pos < 1:
                                hits_at_1_few += 1
                mrr_few += (1.0) / (pos + 1)
            else:
                if pos < 10:
                    hits_at_10_many += 1
                    if pos < 5:
                        hits_at_5_many += 1
                        if pos < 3:
                            hits_at_3_many += 1
                            if pos < 1:
                                hits_at_1_many += 1
                mrr_many += (1.0) / (pos + 1)

    return count,count_few, count_many,\
           hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr, \
           hits_at_1_few, hits_at_3_few, hits_at_5_few, hits_at_10_few, mrr_few, \
           hits_at_1_many, hits_at_3_many, hits_at_5_many, hits_at_10_many, mrr_many
