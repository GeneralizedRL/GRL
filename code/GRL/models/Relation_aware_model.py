import torch.nn as nn
from models.CosNormClassifier import CosNorm_Classifier
from util.utils import *


class MetaEmbedding_Classifier(nn.Module):

    def __init__(self, args):
        super(MetaEmbedding_Classifier, self).__init__()
        num_entities = args.num_entities
        self.represent_mode = args.represent_mode
        self.entity_dim = args.entity_dim
        self.num_entities = args.num_entities
        self.num_relations=args.num_relations
        self.fc_selector = nn.Linear(args.in_dim, 1)
        self.cosnorm_classifier = CosNorm_Classifier(args.in_dim, args.num_relations)
        self.fc2 = nn.Linear(self.entity_dim * 2, self.entity_dim)

    def forward(self, head_entity, tail_entity,relation,all_relation, centroids, openset=False):

        if self.represent_mode == 1:
            self.feat_e1e2 = head_entity - tail_entity
            x = self.feat_e1e2
        elif self.represent_mode == 2:
            self.feat_e1e2 = torch.cat([head_entity, tail_entity], dim=1)
            self.feat_e1e2 = self.fc2(self.feat_e1e2)
            x = self.feat_e1e2
        elif self.represent_mode == 3:
            self.feat_e1e2 = head_entity * tail_entity
            x = self.feat_e1e2

        joint_vector = x
        batch_size = x.size(0)

        x_expand = x.unsqueeze(1).expand(-1, self.num_relations, -1)
        centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids

        similarity_dis = torch.norm(x_expand - centroids_expand, 2, 2)
        similarity_dis=similarity_dis[:, 3:]
        values_nn, labels_nn = torch.sort(similarity_dis, 1)
        attention_value = torch.matmul(x,  all_relation.t())
        attention_value = attention_value.softmax(dim=1)
        one_hot = torch.zeros(batch_size, self.num_relations).cuda().scatter_(1, relation.view(-1, 1), 1).byte()

        attention_value = attention_value.masked_fill(one_hot, torch.tensor(0))
        memory_feature = torch.matmul(attention_value, keys_memory)

        p_fusion = self.fc_selector(x)
        p_fusion = p_fusion.sigmoid()
        x = (torch.tensor(1.0) - p_fusion) * joint_vector + p_fusion * memory_feature
        logits = self.cosnorm_classifier(x)
        if openset:
            return logits, self.feat_e1e2, values_nn[:, 0], labels_nn[:, 0]+torch.tensor(3)
        return logits, self.feat_e1e2


def create_model(args):
    print('Loading Meta Embedding Classifier.')
    args = DottableDict(args)
    clf = MetaEmbedding_Classifier(args)
    return clf
