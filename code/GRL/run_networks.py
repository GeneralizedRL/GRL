'''
The source code references:
[MultiHopKG] Multi-Hop Knowledge Graph Reasoning with Reward Shaping
[OpenLongTailRecognition-OLTR] Large-Scale Long-Tailed Recognition in an Open World
'''

import copy
import os
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from args import args
from util.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class model():
    def __init__(self, config, data, test=False, args=None, kg=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.training_opt = self.config['training_opt']
        self.data = data
        self.test_mode = test
        self.label_smoothing_epsilon = self.training_opt['label_smoothing_epsilon']
        self.grad_norm = self.training_opt['grad_norm']
        self.few_prob = args.few_prob
        self.args = args
        self.kg = kg
        self.init_models(args=args)

        if not self.test_mode:
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers()
            self.init_criterions()

        # Set up log file
        newlogname = change_log_dir(self.training_opt, args)
        self.log_file = os.path.join(self.training_opt['log_dir'], newlogname)

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)

        print_str = [self.config]
        print_write2(print_str, self.log_file)

    def init_models(self, args=None):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []
        print("Using", torch.cuda.device_count(), "GPUs.")
        for key, val in networks_defs.items():
            # Networks
            def_file = val['def_file']
            model_args = val['params']
            model_args['test_mode'] = self.test_mode
            model_args['num_entities'] = len(self.data['train'].dataset.entity2id.keys())
            model_args['num_relations'] = len(self.data['train'].dataset.relation2id.keys())
            model_args['represent_mode'] = args.represent_mode
            model_args['embmodel'] = args.embmodel
            self.embmodel = args.embmodel
            self.networks[key] = source_import(def_file).create_model(model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)

    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}
        optim_params_list = []
        for key, val in criterion_defs.items():
            def_file = val['def_file']
            self.criterions[key] = source_import(def_file).create_loss().to(self.device)
            self.criterion_weights[key] = val['weight']

    def init_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.networks['emb_model'].parameters(), 'lr': args.lr},
            {'params': self.networks['relation_aware_model'].parameters(), 'lr': args.lr}],
            lr=args.lr, )
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.scheduler_params['step_size'],
                                              gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def batch_forward(self, inputs, phase='train', openset=False):
        if openset:
            self.head_entity, self.tail_entity, self.relation, S, self.all_relation_emb = \
                self.networks['emb_model'](inputs, openset=True)
            self.centroids = self.networks['emb_model'].module.relation_embeddings.weight
            _, _, values_sort, label_sort = self.networks['relation_aware_model'](
                self.head_entity, self.tail_entity, torch.split(inputs, 1, 1)[2], self.all_relation_emb,
                self.centroids, openset=True)
            inputs1 = torch.split(inputs, 1, 1)[0]
            inputs2 = torch.split(inputs, 1, 1)[1]
            inputs_ = torch.cat((inputs1, inputs2, label_sort.view(-1, 1)), 1)

            _, _, _, self.score_open, _ = self.networks['emb_model'](inputs_, openset=True)

            with open(self.open_result, 'a') as f:
                for i in range(inputs.size(0)):
                    print_str = [str(self.kg.id2entity[torch.split(inputs, 1, 1)[0][i].item()]) + "#" + str(
                        self.kg.id2entity[torch.split(inputs, 1, 1)[1][i].item()]) + "#" +
                                 str(self.kg.id2relation[torch.split(inputs, 1, 1)[2][i].item()]) + "#" +
                                 str(self.score_open[i].item()) + "#" + str(values_sort[i].item()) + "#" + str(
                        self.kg.id2relation[label_sort[i].item()])]
                    print(*print_str, file=f)

        else:
            if phase == 'train':
                self.S_single, self.features, self.head_entity, self.tail_entity, self.relation, self.all_relation_emb = \
                    self.networks['emb_model'](inputs)

                self.centroids = self.networks['emb_model'].module.relation_embeddings.weight

                self.logits, self.feat_e1e2 = self.networks['relation_aware_model'](self.head_entity,
                                                                                    self.tail_entity,
                                                                                    torch.split(inputs, 1, 1)[2],
                                                                                    self.all_relation_emb,
                                                                                    self.centroids)
            else:
                self.features, self.S_all, self.head_entity, self.tail_entity, self.relation, self.all_entity_emb = \
                    self.networks['emb_model'](inputs)

                self.logits = self.S_all

    def batch_backward(self):
        self.model_optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        # Step optimizers
        if self.grad_norm > 0:
            clip_grad_norm_(self.networks['emb_model'].parameters(), self.grad_norm)
        self.model_optimizer.step()

    def batch_loss(self, labels=None, r_labels=None):
        self.Score_Aware_Loss = self.criterions['Score_Aware_Loss'](self.features, labels) * self.criterion_weights[
            'Score_Aware_Loss']
        self.Relation_Aware_Loss = self.criterions['Relation_Aware_Loss'](self.logits, r_labels) *0.1* \
                                   self.criterion_weights[
                                       'Relation_Aware_Loss']

        self.loss = self.Score_Aware_Loss + self.Relation_Aware_Loss
        print_str = ['\n#total_loss#%.5f' % (self.loss.item()) + '#Loss-conve#%.5f' % (
            self.Score_Aware_Loss.item()) + '#Loss-meta#%.5f' % (self.Relation_Aware_Loss.item()) + '\n']
        return print_str

    def print_all_model_parameters(self, network_model):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in network_model.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in network_model.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()
        print_write2(param_sizes, self.log_file)
        print_write2([sum(param_sizes)], self.log_file)

    def train(self):
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        # Initialize best model
        best_model_weights = {}
        best_model_weights['emb_model'] = copy.deepcopy(self.networks['emb_model'].state_dict())
        best_model_weights['relation_aware_model'] = copy.deepcopy(
            self.networks['relation_aware_model'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            print_str = ["epoch:" + str(epoch)]
            print_write(print_str, self.log_file)
            for model in self.networks.values():
                model.train()
            torch.cuda.empty_cache()
            self.model_optimizer_scheduler.step()
            print("lr1=", self.model_optimizer.param_groups[0]['lr'])
            step = 0
            try:
                with tqdm(self.data['train']) as t:
                    for inputs, labels, _ in t:
                        if step == self.epoch_steps:
                            break
                        step += 1
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        with torch.set_grad_enabled(True):

                            labels = labels.float()
                            labels = ((1 - self.label_smoothing_epsilon) * labels) + (1.0 / labels.size(1))

                            r_labels = []
                            for triple in inputs:
                                r_labels.append(triple[2])
                            r_labels = torch.Tensor(r_labels).view(-1).cuda().long()

                            self.batch_forward(inputs,
                                               phase='train')

                            loss_str = self.batch_loss(labels=labels, r_labels=r_labels)
                            if step % 10 == 0:
                                print_write2(loss_str, self.log_file)
                            self.batch_backward()
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            print_str = str('Epoch=' + str(epoch) + ', Loss=' + str(self.loss.item()))
            print_write(print_str, self.log_file)

            self.eval_batch(phase='dev')

            if self.MRR_ALL > best_acc:
                best_epoch = copy.deepcopy(epoch)
                best_acc = copy.deepcopy(self.MRR_ALL)
                best_model_weights['emb_model'] = copy.deepcopy(self.networks['emb_model'].state_dict())

                best_model_weights['relation_aware_model'] = copy.deepcopy(
                    self.networks['relation_aware_model'].state_dict())
                print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
                print_write(print_str, self.log_file)
                self.save_model(epoch, best_epoch, best_model_weights, best_acc)

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        print()
        print('Training Complete.')
        print('Done')

    def eval_batch(self, phase='dev', openset=False):
        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        if openset:
            print('Under openset test mode')
            self.open_result = os.path.join(self.training_opt['log_dir'], "open_result")
            if os.path.isfile(self.open_result):
                os.remove(self.open_result)
            with open(self.open_result, 'a') as f:
                print_str = ["head_entity#head_entity#relation#score#min_dist#similar_r#"]
                print(*print_str, file=f)

        torch.cuda.empty_cache()
        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        MRR_ALL, HITS_1, HITS_3, HITS_5, HITS_10 = 0.0, 0.0, 0.0, 0.0, 0.0
        MRR_many, HITS_1_many, HITS_3_many, HITS_5_many, HITS_10_many = 0.0, 0.0, 0.0, 0.0, 0.0
        MRR_few, HITS_1_few, HITS_3_few, HITS_5_few, HITS_10_few = 0.0, 0.0, 0.0, 0.0, 0.0

        num_of_samples = 0
        num_of_few = 0
        num_of_many = 0
        num_all = 0

        dev_data = self.data[phase].dataset
        for inputs, labels, paths in tqdm(self.data[phase], ncols=10):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                # In validation or testing
                num_of_samples += inputs.size(0)
                self.batch_forward(inputs, phase=phase, openset=openset)
                if openset:
                    continue
                total_logits = self.logits

                count, count_few, count_many, hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr, \
                hits_at_1_few, hits_at_3_few, hits_at_5_few, hits_at_10_few, mrr_few, \
                hits_at_1_many, hits_at_3_many, hits_at_5_many, hits_at_10_many, mrr_many = hits_and_ranks_all \
                    (inputs, total_logits, dev_data.all_objects,
                     int(len(self.data['train'].dataset.relation2id.keys()) * self.few_prob),
                     rel2e2_notconsider=dev_data.rel2e2_notconsider, openset=openset)

                num_all += count
                HITS_1 += hits_at_1
                HITS_3 += hits_at_3
                HITS_5 += hits_at_5
                HITS_10 += hits_at_10
                MRR_ALL += mrr

                num_of_few += count_few
                HITS_1_few += hits_at_1_few
                HITS_3_few += hits_at_3_few
                HITS_5_few += hits_at_5_few
                HITS_10_few += hits_at_10_few
                MRR_few += mrr_few

                num_of_many += count_many
                HITS_1_many += hits_at_1_many
                HITS_3_many += hits_at_3_many
                HITS_5_many += hits_at_5_many
                HITS_10_many += hits_at_10_many
                MRR_many += mrr_many

        if openset:
            return 0

        self.MRR_ALL = MRR_ALL / num_of_samples
        self.HITS_1 = float(HITS_1) / num_of_samples
        self.HITS_3 = float(HITS_3) / num_of_samples
        self.HITS_5 = float(HITS_5) / num_of_samples
        self.HITS_10 = float(HITS_10) / num_of_samples

        self.MRR_few = MRR_few / num_of_few
        self.HITS_1_few = float(HITS_1_few) / num_of_few
        self.HITS_3_few = float(HITS_3_few) / num_of_few
        self.HITS_5_few = float(HITS_5_few) / num_of_few
        self.HITS_10_few = float(HITS_10_few) / num_of_few

        if num_of_many > 0:
            self.MRR_many = MRR_many / num_of_many
            self.HITS_1_many = float(HITS_1_many) / num_of_many
            self.HITS_3_many = float(HITS_3_many) / num_of_many
            self.HITS_5_many = float(HITS_5_many) / num_of_many
            self.HITS_10_many = float(HITS_10_many) / num_of_many
        else:
            self.MRR_many = 0
            self.HITS_1_many = 0
            self.HITS_3_many = 0
            self.HITS_5_many = 0
            self.HITS_10_many = 0

        print("num_all:", num_all, "num_of_samples:", num_of_samples, "num_of_few:", num_of_few, "num_of_many:",
              num_of_many)
        print_str = ['\n#MRR_ALL#%.5f' % (self.MRR_ALL) + '#HITS_1#%.5f' % (
            self.HITS_1) + '#HITS_3#%.5f' % (self.HITS_3) + '#HITS_5#%.5f' % (
                         self.HITS_5) + '#HITS_10#%.5f' % (self.HITS_10),
                     '\n'
                     + '#MRR_few#%.5f' % (self.MRR_few) + '#HITS_1_few#%.5f' % (
                         self.HITS_1_few) + '#HITS_3_few#%.5f' % (self.HITS_3_few) + '#HITS_5_few#%.5f' % (
                         self.HITS_5_few) + '#HITS_10_few#%.5f' % (self.HITS_10_few),
                     '\n'
                     + '#MRR_many#%.5f' % (self.MRR_many) + '#HITS_1_many#%.5f' % (
                         self.HITS_1_many) + '#HITS_3_many#%.5f' % (self.HITS_3_many) + '#HITS_5_many#%.5f' % (
                         self.HITS_5_many) + '#HITS_10_many#%.5f' % (self.HITS_10_many),
                     '\n'
                     ]
        if phase == 'dev' or phase == 'test':
            print_write(print_str, self.log_file)

    def load_model(self):
        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')

        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)
        model_state = checkpoint['state_dict_best']

        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc):
        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict_best': best_model_weights,
                        'best_acc': best_acc}
        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')
        torch.save(model_states, model_dir)
