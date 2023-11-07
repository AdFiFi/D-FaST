from data import DataConfig


class BaseConfig:
    def __init__(self,
                 d_model=512,
                 d_hidden=512,
                 node_size=200,
                 dim_feedforward=512,
                 node_feature_size=200,
                 time_series_size=100,
                 activation='gelu',
                 dropout=0.1,
                 num_classes=2,
                 num_sites=19,
                 initializer=None,
                 dict_output=True,
                 label_smoothing=0
                 ):
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.node_size = node_size
        self.dim_feedforward = dim_feedforward
        self.time_series_size = time_series_size
        self.node_feature_size = node_feature_size
        self.activation = activation
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_sites = num_sites
        self.initializer = initializer
        self.dict_output = dict_output
        self.label_smoothing = label_smoothing
        self.class_weight = [1]*num_classes

