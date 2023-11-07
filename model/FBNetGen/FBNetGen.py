import torch.nn.functional as F
from .FBNetGenLayers import *
from ..base import BaseConfig, ModelOutputs


class FBNetGenConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 activation='gelu',
                 dropout=0.5,
                 extractor_type='gru',  # gru or cnn
                 d_model=16,
                 num_classes=2,
                 window_size=4,
                 cnn_pool_size=16,
                 graph_generation='product',  # product or linear
                 num_gru_layers=4,
                 group_loss=True,
                 sparsity_loss=True,
                 sparsity_loss_weight=1.0e-4
                 ):
        super(FBNetGenConfig, self).__init__(d_model=d_model,
                                             node_size=node_size,
                                             node_feature_size=node_feature_size,
                                             time_series_size=time_series_size,
                                             activation=activation,
                                             dropout=dropout,
                                             num_classes=num_classes)
        self.extractor_type = extractor_type
        self.window_size = window_size
        self.cnn_pool_size = cnn_pool_size
        self.graph_generation = graph_generation
        self.num_gru_layers = num_gru_layers
        self.group_loss = group_loss
        self.sparsity_loss = sparsity_loss
        self.sparsity_loss_weight = sparsity_loss_weight


class FBNetGen(nn.Module):
    """
    A Reproduction of FBNetGen:
    FBNETGEN: Task-aware GNN-based fMRI Analysis via Functional Brain Network Generation

    X. Kan, H. Cui, J. D. Lukemire, Y. Guo and C. Yang

    International Conference on Medical Imaging with Deep Learning

    """
    def __init__(self, config: FBNetGenConfig):
        super().__init__()

        assert config.extractor_type in ['cnn', 'gru']
        assert config.graph_generation in ['linear', 'product']
        assert config.time_series_size % config.window_size == 0

        self.graph_generation = config.graph_generation
        if config.extractor_type == 'cnn':
            self.extract = ConvKRegion(
                out_size=config.d_model, kernel_size=config.window_size,
                time_series=config.time_series_size)
        elif config.extractor_type == 'gru':
            self.extract = GruKRegion(
                out_size=config.d_model, kernel_size=config.window_size,
                layers=config.num_gru_layers)
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                config.d_model, node_size=config.node_size)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                config.d_model, node_size=config.node_size)

        self.predictor = GNNPredictor(
            config.node_feature_size, node_size=config.node_size, num_classes=config.num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, time_series, node_feature, labels):
        x = self.extract(time_series)               # [batch_sz, node_sz, embedding_size]
        x = F.softmax(x, dim=-1)                    # [batch_sz, node_sz, embedding_size]
        m = self.emb2graph(x)                       # [batch_sz, node_sz, node_sz, 1]
        m = m[:, :, :, 0]                           # [batch_sz, node_sz, node_sz]

        logits = self.predictor(m, node_feature)
        loss = self.loss_fn(logits, labels)
        return ModelOutputs(logits=logits,
                            loss=loss,
                            hidden_state=m)

    def loss(self, logits, label):
        return self.loss_fn(logits, label)
