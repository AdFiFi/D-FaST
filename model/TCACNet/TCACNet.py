from ..base import BaseConfig, ModelOutputs
from .TCACNetLayers import *


class TCACNetModelOutput(ModelOutputs):
    def __init__(self,
                 logits=None,
                 loss=None,
                 hidden_state=None,
                 loss_global_model=None,
                 loss_local_and_top=None
                 ):
        super().__init__(logits=logits,
                         loss=loss,
                         hidden_state=hidden_state)
        self.loss_global_model = loss_global_model
        self.loss_local_and_top = loss_local_and_top


class TCACNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 n_slices=1,
                 alpha=0.05,
                 num_kernels=40):
        super(TCACNetConfig, self).__init__(node_size=node_size,
                                            node_feature_size=node_feature_size,
                                            time_series_size=time_series_size,
                                            num_classes=num_classes)
        self.n_slices = n_slices
        self.alpha = alpha
        self.num_kernels = num_kernels


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, logits):
        b = F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
        b = -1.0 * b.sum()
        return b


class HintLoss(nn.Module):
    def __init__(self, config: TCACNetConfig):
        super(HintLoss, self).__init__()
        self.alpha = config.alpha
        self.n_slices = config.n_slices

    def forward(self, global_feature, local_feature):
        batch_size = global_feature.shape[0]
        raw_hint_loss = torch.sum((global_feature - local_feature).pow(2), dim=0)
        hint_loss = 0.05 * raw_hint_loss / (batch_size * self.n_slices)
        return hint_loss


class TCACNet(nn.Module):
    """
    A Reproduction of TCACNet:
    Mostly based on https://github.com/LiuXiaolin-lxl/TCACNet.git

    TCACNet: Temporal and channel attention convolutional network for motor imagery classification of EEG-based BCI

    X. Liu, R. Shi, Q. Hui, S. Xu, S. Wang, R. Na, et al.

    Information Processing & Management 2022 Vol. 59 Issue 5 Pages 103001
    """

    def __init__(self, config: TCACNetConfig):
        super(TCACNet, self).__init__()
        self.config = config
        self.n_slices = config.n_slices
        self.local_network = LocalNetwork(config)
        self.global_network = GlobalNetwork(config)
        self.top_layer = TopNetwork(config)
        self.entropy_loss_fn = EntropyLoss()
        self.hint_loss = HintLoss(config)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, node_feature, labels):
        time_series = time_series.unsqueeze(1)
        node_feature = node_feature.unsqueeze(1)
        with torch.enable_grad():
            conv2_weight = self.global_network.conv2.weight.reshape(-1, self.config.node_size)
            conv2_weight_abs = torch.abs(conv2_weight)
            channel_w = conv2_weight_abs.mean(dim=0)
            channel_per = node_feature
            channel_per_m = channel_per.mean(dim=0)
            channel_per_m = channel_per_m[0, :]
            two_22 = torch.full((self.config.node_size,), 2.0)
            two_ex = torch.pow(two_22.to(time_series.device), 1 / channel_per_m)
            channel_loss = 0.005 * torch.sum(two_ex * channel_w, dim=0)

            global_features = self.global_network(time_series)
            logits1 = self.top_layer(global_features)
            entropy = self.entropy_loss_fn(logits1)
            d_ent_d_features = torch.autograd.grad(entropy, global_features,
                                                   grad_outputs=torch.ones(entropy.size(), device=time_series.device))
            d_entropy_d_features = torch.cat(d_ent_d_features, 0)

        top_k_indices = self.salient_indices(d_entropy_d_features, time_series.shape[0])
        _, _, _, feature_w = global_features.shape
        local_features, local_indexes = self.extract_features(time_series, top_k_indices, feature_w)
        features, flat_global_replaced, flat_local_features = self.replace_features(global_features, local_features,
                                                                                    local_indexes)

        logits2 = self.top_layer(features)
        hint_loss = self.hint_loss(flat_global_replaced, flat_local_features)
        loss_local_and_top = self.loss_fn(logits2, labels)
        loss_global_model = loss_local_and_top + hint_loss + channel_loss
        if self.config.dict_output:
            return TCACNetModelOutput(logits=logits2,
                                      loss=loss_global_model,
                                      loss_local_and_top=loss_local_and_top,
                                      loss_global_model=loss_global_model)
        else:
            return logits2, loss_local_and_top, loss_global_model

    def salient_indices(self, grads, batch_size):
        M = torch.sqrt((torch.sum(-grads, dim=1)).pow(2))
        M_reshape = M.view(batch_size, -1)
        values, indices = torch.topk(M_reshape, self.n_slices, largest=False)
        assert indices.shape == (batch_size, self.n_slices)
        return indices

    def extract_features(self, time_series, k_indexes, feature_w):
        batch_size, _, _, _ = time_series.shape
        k_src_idxs = []
        k_slices = []
        for i in range(self.n_slices):
            src_idxs, slices, left_coord = self.extract_one_feature(time_series, k_indexes[:, i], feature_w)
            k_src_idxs.append(src_idxs)
            k_slices.append(slices)
        concat_slices = torch.cat(k_slices, 0).to(time_series.device)
        assert concat_slices.shape == (batch_size * self.n_slices, 1, self.config.node_size, 78)
        concat_k_features = self.local_network(concat_slices)
        k_features = torch.split(concat_k_features, self.n_slices, 0)
        return k_features, k_src_idxs

    def extract_one_feature(self, time_series, indexes, feature_w):
        left_coord, idx_i, idx_j = self.left_coordinates(indexes, feature_w, time_series.device)
        slices = self.extract_slices(time_series, left_coord)
        src_idxs = torch.cat((idx_i.view(-1, 1), idx_j.view(-1, 1)), 1).to(time_series.device)
        return src_idxs, slices, left_coord

    def extract_slices(self, time_series, left_coord):
        batch_size = time_series.shape[0]
        slices = []
        for i in range(batch_size):
            sliced = time_series[i].narrow(1, left_coord[i, 0], self.config.node_size)
            sliced = sliced.narrow(2, left_coord[i, 1], 78)
            slices.append(sliced)
        slices = torch.stack(slices).to(time_series.device)
        return slices

    @staticmethod
    def left_coordinates(indexes, feature_w, device):
        indexes = indexes.long().to(device)
        idx_x = indexes // feature_w
        idx_y = torch.fmod(indexes, feature_w)
        x_left = idx_x.view(-1, 1)
        y_left = (40 * idx_y).view(-1, 1)
        coords = torch.cat((x_left, y_left), 1)
        return coords, idx_x, idx_y

    @staticmethod
    def replace_features(global_features, local_features, replace_idxs):
        batch_size, feature_ch, feature_h, feature_w = global_features.size()

        def _convert_to_1d_idxs(src_idxs):
            batch_idx_len = feature_ch * feature_w * feature_h
            batch_idx_base = (torch.Tensor([i * batch_idx_len for i in range(batch_size)]).long()).to(
                global_features.device)
            batch_1d = feature_ch * feature_w * src_idxs[:, 0] + feature_ch * src_idxs[:, 1]
            batch_1d = torch.add(batch_1d, batch_idx_base)
            flat_idxs = [batch_1d + i for i in range(feature_ch)]
            flat_idxs = (torch.stack(flat_idxs)).t()
            flat_idxs = flat_idxs.contiguous().view(-1)
            return flat_idxs

        flat_global_features = global_features.view(-1)
        flat_local_features = [i.view(-1) for i in local_features]
        flat_local_features = torch.cat(flat_local_features, 0)
        flat_local_idxs = [_convert_to_1d_idxs(i) for i in replace_idxs]
        flat_local_idxs = torch.cat(flat_local_idxs, 0)
        flat_global_replaced = torch.gather(flat_global_features, 0, flat_local_idxs)
        if flat_global_replaced.size() != flat_local_features.size():
            print('Assertion error : flat_global_replaced.size()', flat_global_replaced.size(),
                  ' !=  flat_local_features.size()', flat_local_features.size())
            assert flat_global_replaced.size() == flat_local_features.size()
        merged = flat_global_features
        merged.clone()[flat_local_idxs] = flat_local_features
        merged = merged.view(global_features.size())

        return merged, flat_global_replaced, flat_local_features
