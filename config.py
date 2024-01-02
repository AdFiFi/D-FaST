from model import *
import torch.nn as nn
from data import DataConfig
import argparse


def init_model_config(args, data_config: DataConfig):
    if args.model == "BNT":
        model_config = BNTConfig(node_size=data_config.node_size,
                                 sizes=(data_config.node_size, data_config.node_size // 2),
                                 num_classes=data_config.num_class,
                                 pooling=(False, True),
                                 pos_encoding=None,  # identity, none
                                 orthogonal=True,
                                 # freeze_center=True,
                                 freeze_center=False,
                                 project_assignment=True,
                                 num_heads=args.num_heads,
                                 pos_embed_dim=data_config.node_size,
                                 dim_feedforward=1024,
                                 )
        model = BNT(model_config)
    elif args.model == "FBNetGen":
        model_config = FBNetGenConfig(activation='gelu',
                                      dropout=0.5,
                                      # extractor_type='gru',  # gru or cnn
                                      extractor_type='cnn',  # gru or cnn
                                      # d_model=16,
                                      d_model=40,
                                      node_size=data_config.node_size,
                                      node_feature_size=data_config.node_feature_size,
                                      time_series_size=data_config.time_series_size,
                                      num_classes=data_config.num_class,
                                      window_size=5,
                                      # window_size=40,
                                      # window_size=50,
                                      cnn_pool_size=16,
                                      graph_generation='product',  # product or linear
                                      num_gru_layers=4,
                                      group_loss=True,
                                      sparsity_loss=True,
                                      sparsity_loss_weight=1.0e-4)
        model = FBNetGen(model_config)
    elif args.model == 'BrainNetCNN':
        model_config = BrainNetCNNConfig(node_size=data_config.node_size,
                                         num_classes=data_config.num_class)
        model = BrainNetCNN(model_config)
    elif args.model == 'STAGIN':
        model_config = STAGINConfig(node_size=data_config.node_size,
                                    num_classes=data_config.num_class,
                                    d_model=args.d_model,
                                    num_layers=args.num_layers,
                                    window_size=args.window_size,
                                    window_stride=args.window_stride,
                                    dynamic_length=args.dynamic_length,
                                    sampling_init=args.sampling_init)
        model = STAGIN(model_config)
    elif args.model == "Transformer":
        model_config = TransformerConfig(node_size=data_config.node_size,
                                         num_classes=data_config.num_class,
                                         node_feature_size=data_config.node_feature_size,
                                         readout='concat',
                                         num_layers=args.num_layers)
        model = Transformer(model_config)
    elif args.model == "EEGNet":
        model_config = EEGNetConfig(node_size=data_config.node_size,
                                    time_series_size=data_config.time_series_size,
                                    node_feature_size=data_config.node_feature_size,
                                    num_classes=data_config.num_class,
                                    frequency=args.frequency,
                                    D=args.D,
                                    num_kernels=args.num_kernels,
                                    p1=args.p1,
                                    p2=args.p2,
                                    dropout=args.dropout)
        model_config.class_weight = data_config.class_weight
        model = EEGNet(model_config)
    elif args.model == "DFaST":
        model_config = DFaSTConfig(node_size=data_config.node_size,
                                   time_series_size=data_config.time_series_size,
                                   node_feature_size=data_config.node_feature_size,
                                   num_classes=data_config.num_class,
                                   sparsity=args.sparsity,
                                   frequency=args.frequency,
                                   D=args.D,
                                   p1=args.p1,
                                   p2=args.p2,
                                   k=args.k,
                                   num_kernels=args.num_kernels,
                                   d_model=args.d_model,
                                   window_size=args.window_size,
                                   window_stride=args.window_stride,
                                   dynamic_length=args.dynamic_length,
                                   num_heads=args.num_heads,
                                   dim_feedforward=args.dim_feedforward,
                                   num_spatial_layers=args.num_layers,
                                   num_node_temporal_layers=args.num_node_temporal_layers,
                                   num_graph_temporal_layers=args.num_graph_temporal_layers,
                                   attention_depth=args.attention_depth,
                                   activation=args.activation,
                                   dropout=args.dropout,
                                   # distill=(False, ) + (args.num_layers - 1) *
                                   #         ((True,) if args.distill else (False,)),
                                   distill=args.num_layers * ((True,) if args.distill else (False,)),
                                   initializer=args.initializer,
                                   label_smoothing=args.epsilon_ls
                                   )
        model_config.class_weight = data_config.class_weight
        model = DFaSTForClassification(model_config)
    elif args.model == "DFaSTOnlySpatial":
        model_config = DFaSTConfig(node_size=data_config.node_size,
                                   time_series_size=data_config.time_series_size,
                                   node_feature_size=data_config.node_feature_size,
                                   num_classes=data_config.num_class,
                                   sparsity=args.sparsity,
                                   frequency=args.frequency,
                                   D=args.D,
                                   p1=args.p1,
                                   p2=args.p2,
                                   k=args.k,
                                   num_kernels=args.num_kernels,
                                   d_model=args.d_model,
                                   window_size=args.window_size,
                                   window_stride=args.window_stride,
                                   dynamic_length=args.dynamic_length,
                                   num_heads=args.num_heads,
                                   dim_feedforward=args.dim_feedforward,
                                   num_spatial_layers=args.num_layers,
                                   num_node_temporal_layers=args.num_node_temporal_layers,
                                   num_graph_temporal_layers=args.num_graph_temporal_layers,
                                   attention_depth=args.attention_depth,
                                   activation=args.activation,
                                   dropout=args.dropout,
                                   # distill=(False, ) + (args.num_layers - 1) *
                                   #         ((True,) if args.distill else (False,)),
                                   distill=args.num_layers * ((True,) if args.distill else (False,)),
                                   initializer=args.initializer,
                                   label_smoothing=args.epsilon_ls
                                   )
        model_config.class_weight = data_config.class_weight
        model = DFaSTOnlySpatialForClassification(model_config)
    elif args.model == "LMDA":
        model_config = LMDAConfig(node_size=data_config.node_size,
                                  time_series_size=data_config.time_series_size,
                                  node_feature_size=data_config.node_feature_size,
                                  num_classes=data_config.num_class,
                                  depth=9,
                                  channel_depth1=args.num_kernels,
                                  channel_depth2=9,
                                  ave_depth=1,
                                  avepool=5
                                  )
        model_config.class_weight = data_config.class_weight
        model = LMDA(model_config)
    elif args.model == "ShallowConvNet":
        model_config = ShallowConvNetConfig(node_size=data_config.node_size,
                                            time_series_size=data_config.time_series_size,
                                            node_feature_size=data_config.node_feature_size,
                                            num_classes=data_config.num_class,
                                            num_kernels=args.num_kernels
                                            )
        model_config.class_weight = data_config.class_weight
        model = ShallowConvNet(model_config)
    elif args.model == "DeepConvNet":
        model_config = DeepConvNetConfig(node_size=data_config.node_size,
                                         time_series_size=data_config.time_series_size,
                                         node_feature_size=data_config.node_feature_size,
                                         num_classes=data_config.num_class,
                                         num_kernels=25
                                         )
        model_config.class_weight = data_config.class_weight
        model = DeepConvNet(model_config)
    elif args.model == "RACNN":
        model_config = RACNNConfig(node_size=data_config.node_size,
                                   time_series_size=data_config.time_series_size,
                                   node_feature_size=data_config.node_feature_size,
                                   num_classes=data_config.num_class,
                                   k=args.k
                                   )
        model_config.class_weight = data_config.class_weight
        model = RACNN(model_config)
    elif args.model == "EEGChannelNet":
        model_config = EEGChannelNetConfig(node_size=data_config.node_size,
                                           time_series_size=data_config.time_series_size,
                                           node_feature_size=data_config.node_feature_size,
                                           num_classes=data_config.num_class
                                           )
        model_config.class_weight = data_config.class_weight
        model = EEGChannelNet(model_config)
    elif args.model == "TCANet":
        model_config = TCANetConfig(node_size=data_config.node_size,
                                    time_series_size=data_config.time_series_size,
                                    node_feature_size=data_config.node_feature_size,
                                    num_classes=data_config.num_class
                                    )
        model_config.class_weight = data_config.class_weight
        model = TCANet(model_config)
    elif args.model == "TCACNet":
        model_config = TCACNetConfig(node_size=data_config.node_size,
                                     time_series_size=data_config.time_series_size,
                                     node_feature_size=data_config.node_feature_size,
                                     num_classes=data_config.num_class
                                     )
        model_config.class_weight = data_config.class_weight
        model = TCACNet(model_config)
    elif args.model == "SBLEST":
        model_config = SBLESTConfig(node_size=data_config.node_size,
                                    time_series_size=data_config.time_series_size,
                                    node_feature_size=data_config.node_feature_size,
                                    num_classes=data_config.num_class
                                    )
        model_config.class_weight = data_config.class_weight
        model = SBLEST(model_config)
    else:
        model = None
        model_config = None
    if model is not None:
        init_parameters(model, model_config)
    return model, model_config


def init_parameters(model, model_config):
    if model_config.initializer is not None:
        for p in model.parameters():
            if p.dim() > 1:
                if model.config.initializer == 'xavier':
                    nn.init.xavier_uniform_(p)
                elif model.config.initializer == 'orthogonal':
                    nn.init.orthogonal_(p)


def init_config():
    parser = argparse.ArgumentParser()

    global_group = parser.add_argument_group(title="global", description="")
    global_group.add_argument("--project", default="", type=str, help="")
    global_group.add_argument("--wandb_entity", default='cwg', type=str, help="")
    global_group.add_argument("--log_dir", default="./log_dir", type=str, help="")
    global_group.add_argument("--model", default="FaSTP", type=str, help="")
    global_group.add_argument("--num_repeat", default=5, type=int, help="")
    global_group.add_argument("--within_subject", action="store_true", help="")
    global_group.add_argument("--subject_num", default=10, type=int, help="")
    global_group.add_argument("--visualize", action="store_true", help="")

    data_group = parser.add_argument_group(title="data", description="")
    data_group.add_argument("--dataset", default='BR', type=str, help="")
    data_group.add_argument("--data_dir", default="", type=str, help="")
    data_group.add_argument("--data_processors", default=0, type=int, help="")
    data_group.add_argument("--train_set", default=0.7, type=float, help="")
    data_group.add_argument("--val_set", default=0.1, type=float, help="")
    data_group.add_argument("--percentage", default=1., type=float, help="")
    data_group.add_argument("--batch_size", default=64, type=int, help="")
    data_group.add_argument("--num_epochs", default=200, type=int, help="")
    data_group.add_argument("--drop_last", default=True, type=bool, help="")
    data_group.add_argument("--dynamic", action="store_true", help="")
    data_group.add_argument("--subject_id", default=0, type=int, help="")
    data_group.add_argument("--frequency", default=500, type=int, help="")
    data_group.add_argument("--D", default=2, type=int, help="")
    data_group.add_argument("--F1", default=8, type=int, help="")
    data_group.add_argument("--p1", default=4, type=int, help="")
    data_group.add_argument("--p2", default=8, type=int, help="")

    preprocess_group = parser.add_argument_group(title="preprocess", description="")
    preprocess_group.add_argument("--mix_up", action="store_true", help="")

    model_group = parser.add_argument_group(title="model", description="")
    model_group.add_argument("--d_model", default=4, type=int, help="")
    model_group.add_argument("--k", default=5, type=int, help="")
    model_group.add_argument("--num_kernels", default=5, type=int, help="")
    model_group.add_argument("--sparsity", default=0.7, type=float, help="")
    model_group.add_argument("--window_size", default=50, type=int, help="")
    model_group.add_argument("--window_stride", default=3, type=int, help="")
    model_group.add_argument("--dynamic_length", default=600, type=int, help="")
    model_group.add_argument("--dynamic_stride", default=1, type=int, help="")
    model_group.add_argument("--dim_feedforward", default=1024, type=int, help="")
    model_group.add_argument("--sampling_init", default=None, type=int, help="")
    model_group.add_argument("--hidden_dim", default=1024, type=int, help="")
    model_group.add_argument("--num_heads", default=1, type=int, help="")
    model_group.add_argument("--num_layers", default=2, type=int, help="")
    model_group.add_argument("--num_node_temporal_layers", default=1, type=int, help="")
    model_group.add_argument("--num_graph_temporal_layers", default=2, type=int, help="")
    model_group.add_argument("--attention_depth", default=2, type=int, help="")
    model_group.add_argument("--activation", default="gelu", type=str, help="")
    model_group.add_argument("--model_dir", default="output_dir", type=str, help="")
    model_group.add_argument("--dropout", default=0.5, type=float, help="")
    model_group.add_argument("--distill", action="store_true", help="")
    model_group.add_argument("--initializer", default=None, type=str, help="")

    train_group = parser.add_argument_group(title="train", description="")
    train_group.add_argument("--do_train", action="store_true", help="")
    train_group.add_argument("--do_parallel", action="store_true", help="")
    train_group.add_argument("--device", default="cuda", type=str, help="")
    train_group.add_argument("--save_steps", default=200, type=int, help="")
    train_group.add_argument("--epsilon_ls", default=0, type=float, help=" label_smoothing")
    train_group.add_argument("--alpha", default=1., type=float, help="")
    train_group.add_argument("--beta", default=1., type=float, help="")

    optimizer_group = parser.add_argument_group(title="optimizer", description="")
    optimizer_group.add_argument("--optimizer", default='Adam', type=str, help="")
    optimizer_group.add_argument("--learning_rate", default=1e-4, type=float, help="")
    optimizer_group.add_argument("--target_learning_rate", default=1e-5, type=float, help="")
    optimizer_group.add_argument("--max_learning_rate", default=0.001, type=float, help="")
    optimizer_group.add_argument("--beta1", default=0.9, type=float, help="")
    optimizer_group.add_argument("--beta2", default=0.98, type=float, help="")
    optimizer_group.add_argument("--epsilon", default=1e-9, type=float, help="")
    optimizer_group.add_argument("--schedule", default='cos', type=str, help="")
    optimizer_group.add_argument("--warmup_steps", default=400, type=int, help="")
    optimizer_group.add_argument("--weight_decay", default=1e-4, type=float, help="")
    optimizer_group.add_argument("--no_weight_decay", action="store_true", help="")
    optimizer_group.add_argument("--match_rule", default=None, type=str, help="")
    optimizer_group.add_argument("--except_rule", default=None, type=str, help="")

    evaluate_group = parser.add_argument_group(title="evaluate", description="")
    evaluate_group.add_argument("--do_evaluate", action="store_true", help="")
    evaluate_group.add_argument("--do_test", action="store_true", help="")

    return parser.parse_args()
