import torch
import torch.nn as nn

from .dec import DEC
from ..BrainCube import Layers
from ..base import BaseConfig, ModelOutputs


class BNTConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 sizes=(360, 100),
                 pooling=(False, True),
                 pos_encoding=None,  # identity, none
                 orthogonal=True,
                 freeze_center=True,
                 project_assignment=True,
                 pos_embed_dim=360,
                 dim_feedforward=1024,
                 num_heads=4,
                 ):
        super(BNTConfig, self).__init__(node_size=node_size,
                                        dim_feedforward=dim_feedforward)
        self.sizes = sizes
        self.pooling = pooling
        self.pos_encoding = pos_encoding
        self.orthogonal = orthogonal
        self.freeze_center = freeze_center
        self.project_assignment = project_assignment
        self.pos_embed_dim = pos_embed_dim
        self.num_heads = num_heads
        self.distill = True
        self.attention_depth = 1


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self,
                 config,
                 layer_idx,
                 input_node_num,
                 output_node_num,
                 pooling=True):
        super().__init__()
        self.transformer = Layers.BrainCubeEncoderLayer(config,
                                                        layer_idx,
                                                        node_size=input_node_num,
                                                        node_feature_size=config.node_feature_size)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(config.node_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          config.node_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num,
                           hidden_dimension=config.node_feature_size,
                           encoder=self.encoder,
                           orthogonal=config.orthogonal,
                           freeze_center=config.freeze_center,
                           project_assignment=config.project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x, _ = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BNT(nn.Module):
    """BrainNetworkTransformer"""

    def __init__(self, config: BNTConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        d_model = config.node_size

        self.pos_encoding = config.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.node_size, config.pos_embed_dim), requires_grad=True)
            d_model = config.node_size + config.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.sizes
        in_sizes = [sizes[0], sizes[0]]
        do_pooling = config.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(config,
                                    index,
                                    input_node_num=in_sizes[index],
                                    output_node_num=size,
                                    pooling=do_pooling[index]))

        self.dim_reduction = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        # self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, node_feature: torch.tensor, labels: torch.tensor):

        bz, _, _, = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        logits = self.fc(node_feature)
        loss = self.loss_fn(logits, labels)
        return ModelOutputs(logits=logits,
                            loss=loss)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def kl_loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all

