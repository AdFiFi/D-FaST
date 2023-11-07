
class DataConfig:
    def __init__(self, args, time_series_size=0, node_size=0, node_feature_size=0, num_class=2):
        self.dataset = args.dataset
        self.data_dir = args.data_dir
        self.train_set = args.train_set
        self.percentage = args.percentage
        self.val_set = args.val_set
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.drop_last = args.drop_last
        self.alpha = args.alpha
        self.beta = args.beta
        self.n_splits = args.num_repeat
        self.time_series_size = time_series_size
        self.node_size = node_size
        self.node_feature_size = node_feature_size
        self.num_class = num_class
        self.dynamic = args.dynamic
        self.subject_id = args.subject_id
        self.class_weight = [1]*num_class


# class ABIDEConfig(DataConfig):
#     def __init__(self,
#                  args, ):
#         super(ABIDEConfig, self).__init__(args,
#                                           time_series_sz=100,
#                                           node_sz=200,
#                                           node_feature_sz=200)


