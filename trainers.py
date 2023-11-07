import pywt
from einops import repeat, rearrange
from scipy import signal

from utils import *


class FaSTPTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FaSTPTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                time_series, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "labels": labels.float().to(self.device)}


class FaSTPOnlySpatialTrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FaSTPOnlySpatialTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class FaSPTrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FaSPTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class LMDATrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(LMDATrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class ShallowConvNetTrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(ShallowConvNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class DeepConvNetTrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(DeepConvNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCubeTrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCubeTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCube2Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube2Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                inputs[0], inputs[1], y1=inputs[2].float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
            return {"time_series": time_series.to(self.device),
                    "node_feature": self.process_dynamic(inputs[0]).to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": inputs[0].to(self.device),
                    "node_feature": self.process_dynamic(inputs[0]).to(self.device),
                    "labels": inputs[2].float().to(self.device)}

    def process_dynamic(self, time_series):
        B, N, E = time_series.shape
        # time_series = time_series.reshape(-1, N, E // self.model_config.d_model)
        # node_feature = torch.stack([torch.corrcoef(series) for series in time_series])
        # node_feature = node_feature.reshape(B, self.model_config.d_model, N, N)
        # return node_feature

        starts = list(range(0, E-self.model_config.window_size+1, (E-self.model_config.window_size)
                            // (self.model_config.d_model-1)))
        splits = torch.stack([time_series[:, :, start:start+self.model_config.window_size] for start in starts], dim=1)
        # former = splits[:, :-self.model_config.dynamic_stride, :, :]
        # latter = splits[:, self.model_config.dynamic_stride:, :, :]
        node_feature = torch.stack([corrcoef(series) for series in splits.reshape(-1, N, self.model_config.window_size)])
        return node_feature


class BrainCube3Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube3Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCube4Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube4Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCube5Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube5Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        if self.args.dataset == 'ABIDE':
            if self.model.training and self.args.mix_up:
                time_series, node_feature, labels, site_labels = continues_mixup_data(
                    # inputs[0], inputs[1], y1=inputs[2].float(), y2=inputs[3], alpha=self.data_config.alpha, beta=self.data_config.beta)
                    inputs[0], inputs[1], y1=inputs[2].float(), y2=inputs[3])
                return {"time_series": time_series.to(self.device),
                        "node_feature": node_feature.to(self.device),
                        "labels": labels.to(self.device),
                        "site_labels": site_labels.long().to(self.device)}
            else:
                return {"time_series": inputs[0].to(self.device),
                        "node_feature": inputs[1].to(self.device),
                        "labels": inputs[2].float().to(self.device),
                        "site_labels": inputs[3].long().to(self.device)}
        else:
            if self.model.training and self.args.mix_up:
                time_series, node_feature, labels, _ = continues_mixup_data(
                    # inputs[0], inputs[1], y1=inputs[2].float(), y2=inputs[3], alpha=self.data_config.alpha, beta=self.data_config.beta)
                    inputs[0], inputs[1], y1=inputs[2].float())
                return {"time_series": time_series.to(self.device),
                        "node_feature": node_feature.to(self.device),
                        "labels": labels.to(self.device)}
            else:
                return {"time_series": inputs[0].to(self.device),
                        "node_feature": inputs[1].to(self.device),
                        "labels": inputs[2].float().to(self.device)}


class BrainCube6Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube6Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                inputs[0], inputs[1], y1=inputs[2].float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                # inputs[0], inputs[1], y1=inputs[2].float())
            return {"time_series": time_series.to(self.device),
                    "node_feature": self.process_dynamic(time_series).to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": inputs[0].to(self.device),
                    "node_feature": self.process_dynamic(inputs[0]).to(self.device),
                    "labels": inputs[2].float().to(self.device)}

    def process_dynamic(self, time_series):
        B, N, E = time_series.shape
        # time_series = time_series.reshape(-1, N, E // self.model_config.d_model)
        # node_feature = torch.stack([torch.corrcoef(series) for series in time_series])
        # node_feature = node_feature.reshape(B, self.model_config.d_model, N, N)
        # return node_feature

        # starts = list(range(0, E-self.model_config.window_size+1, self.model_config.window_stride))
        starts = list(range(0, self.model_config.dynamic_length-self.model_config.window_size+1,
                            self.model_config.window_stride))
        splits = torch.stack([time_series[:, :, start:start+self.model_config.window_size] for start in starts], dim=1)
        feature_static = torch.stack(
            [corrcoef(series) for series in splits.contiguous().view(-1, N, self.model_config.window_size)])
        feature_static = rearrange(feature_static, '(b w) n e -> b w n e', b=B)

        # former = splits[:, :-self.model_config.dynamic_stride, :, :].contiguous().view(-1, N, self.model_config.window_size)
        # latter = splits[:, self.model_config.dynamic_stride:, :, :].contiguous().view(-1, N, self.model_config.window_size)
        # couple = torch.cat([former, latter], dim=1)
        # feature_dynamic = torch.stack([torch.corrcoef(series) for series in couple])[:, :N, N:]
        # feature_dynamic = rearrange(feature_dynamic, '(b w) n e -> b w n e', b=B)
        #
        # feature = torch.stack([feature_static[:, :-1], feature_dynamic], dim=2)
        # feature = rearrange(feature, 'b t d n e -> b (t d) n e')
        feature = feature_static
        # # feature = feature_static[:, :-1] + feature_dynamic
        #
        # # feature = torch.stack([torch.arctanh(f) for f in feature])
        # # feature[feature == torch.inf] = 0
        # # diag = torch.diag_embed(torch.diagonal(feature, dim1=1, dim2=2))
        # # feature = feature - diag
        # # feature = feature.reshape(B, -1, N, N)
        return feature


class BrainCube7Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube7Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCube8Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube8Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCube9Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube9Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCube10Trainer(BrainCubeTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCube10Trainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BrainCubeXTrainer(BrainCube5Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainCubeXTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class BNTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BNTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        node_feature = inputs['correlation']
        labels = inputs['labels']

        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}


class FBNetGenTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(FBNetGenTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        time_series_size = time_series.shape[-1] // self.model_config.window_size * self.model_config.window_size
        time_series = time_series[:, :, :time_series_size]
        node_feature = inputs['correlation']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, node_feature, labels, _ = continues_mixup_data(
                time_series, node_feature, y1=labels.float())
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.to(self.device)}
        else:
            return {"time_series": time_series.to(self.device),
                    "node_feature": node_feature.to(self.device),
                    "labels": labels.float().to(self.device)}


class BrainNetCNNTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(BrainNetCNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class TransformerTrainer(BNTTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TransformerTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)


class STAGINTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(STAGINTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs, **kwargs):
        dyn_a, sampling_points = process_dynamic_fc(inputs['time_series'].transpose(1, 2),
                                                    self.model_config.window_size,
                                                    self.model_config.window_stride,
                                                    # self.model_config.dynamic_length,
                                                    self.model_config.sampling_init)
        sampling_endpoints = [p + self.model_config.window_size for p in sampling_points]
        dyn_v = repeat(torch.eye(self.model_config.node_size), 'n1 n2 -> b t n1 n2', t=len(sampling_points),
                       b=self.args.batch_size)
        if len(dyn_a) < self.args.batch_size:
            dyn_v = dyn_v[:len(dyn_a)]
        return {'v': dyn_v.to(self.device),
                'a': dyn_a.to(self.device),
                't': inputs['time_series'].permute(2, 0, 1).to(self.device),
                'sampling_endpoints': sampling_endpoints,
                'labels': inputs['labels'].float().to(self.device)}

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=0)):
            input_kwargs = self.prepare_inputs_kwargs(inputs, step=step)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.model_config.clip_grad > 0.0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.model_config.clip_grad)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            wandb.log({'Training loss': loss.item(),
                       'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        # wandb.watch(self.model)
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch"):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.8 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.2
            self.test_result = self.evaluate()
            self.best_result = self.test_result
            msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)


class EEGNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EEGNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha,
                # beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}


class EEGChannelNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(EEGChannelNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}


class RACNNTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(RACNNTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        # time_series = self.get_complex_morlet_wavelets(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "labels": labels.float().to(self.device)}

    @staticmethod
    def get_complex_morlet_wavelets(time_series):
        time_series = time_series.numpy()
        Fa = np.arange(4, 31)
        new_time_series = []
        for i, ts in enumerate(time_series):
            cwt = abs(pywt.cwt(ts, Fa, 'cmor1-1', 1/200)[0])
            new_time_series.append(torch.from_numpy(cwt))
        time_series = torch.stack(new_time_series, dim=0)
        time_series -= time_series.mean()
        time_series /= time_series.std()
        return time_series


class SBLESTTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(SBLESTTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)
        self.W = None
        self.Wh = None

    def load_datasets(self):
        # datasets = eval(
        #     f"load_{self.args.dataset}_data")(self.data_config)
        datasets = eval(
            f"{self.args.dataset}Dataset")(self.data_config, k=self.task_id, subject_id=self.subject_id)

        if self.args.do_parallel:
            data_loaders = init_distributed_dataloader(self.data_config, datasets)
        else:
            data_loaders = init_StratifiedKFold_dataloader(self.data_config, datasets)
        return data_loaders

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        labels = inputs['labels']
        labels = ((labels[:, 1] == 1) * 2 - 1)
        idx0 = torch.argwhere(labels == -1)
        idx1 = torch.argwhere(labels == 1)
        idx0 = idx0[:len(idx1)]
        idx = torch.concat([idx0, idx1], dim=0).squeeze()
        time_series = time_series[idx]
        labels = labels[idx].unsqueeze(-1)
        time_series = time_series.permute(1, 2, 0)

        return {"time_series": time_series.double().to(self.device),
                "labels": labels.double().to(self.device)}

    def train(self):
        train_dataloader = self.data_loaders['train']
        self.model.eval()
        inputs = {"time_series": torch.DoubleTensor().to(self.device),
                  "labels": torch.DoubleTensor().to(self.device)}
        for batch in train_dataloader:
            input_kwargs = self.prepare_inputs_kwargs(batch)
            inputs["time_series"] = torch.concat([inputs["time_series"], input_kwargs["time_series"]], dim=-1)
            inputs["labels"] = torch.concat([inputs["labels"], input_kwargs["labels"]], dim=0)
            # break

        self.W, alpha, V, self.Wh = self.model(**inputs)
        self.best_result = self.test_result = self.evaluate()
        wandb.log({f"Best {k}": v for k, v in self.best_result.items()})

    def evaluate(self):
        test_dataloader = self.data_loaders['test']
        self.model.eval()
        inputs = {"time_series": torch.DoubleTensor().to(self.device),
                  "labels": torch.DoubleTensor().to(self.device)}
        for batch in test_dataloader:
            input_kwargs = self.prepare_inputs_kwargs(batch)
            inputs["time_series"] = torch.concat([inputs["time_series"], input_kwargs["time_series"]], dim=-1)
            inputs["labels"] = torch.concat([inputs["labels"], input_kwargs["labels"]], dim=0)
        R_test, _ = self.model.enhance_conv(inputs["time_series"], self.Wh)
        vec_W = self.W.T.flatten()  # vec operation (Torch)
        preds = R_test @ vec_W
        result = self.metrix(preds, inputs["labels"])
        wandb.log(result)
        return result

    @staticmethod
    def metrix(predict_Y, Y_test):
        """Compute classification accuracy for test set"""

        predict_Y = predict_Y.cpu().numpy()
        Y_test = torch.squeeze(Y_test).cpu().numpy()
        total_num = len(predict_Y)
        error_num = 0
        auc = roc_auc_score(Y_test*0.5+0.5, predict_Y*0.5+0.5)
        # Compute classification accuracy for test set
        Y_predict = np.zeros(total_num)
        for i in range(total_num):
            if predict_Y[i] > 0:
                Y_predict[i] = 1
            else:
                Y_predict[i] = -1

        # Compute classification accuracy
        for i in range(total_num):
            if Y_predict[i] != Y_test[i]:
                error_num = error_num + 1

        acc = (total_num - error_num) / total_num

        report = classification_report(
            Y_test * 0.5 + 0.5, Y_predict * 0.5 + 0.5, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k, v in report.items():
            if '.' in k:
                recall[int(float(k))] = v['recall']
        specificity = recall[0]
        sensitivity = recall[1]

        result = {"Accuracy": acc,
                  "AUC": auc,
                  "Specificity": specificity,
                  "Sensitivity": sensitivity}
        return result


class TCANetTrainer(FaSTPTrainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TCANetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            for param in self.model.local_network.parameters():
                param.requires_grad = False
            for param in self.model.top_layer.parameters():
                param.requires_grad = False
            loss_global_model.backward(retain_graph=True)

            for param in self.model.local_network.parameters():
                param.requires_grad = True
            for param in self.model.top_layer.parameters():
                param.requires_grad = True
            for param in self.model.global_network.parameters():
                param.requires_grad = False
            loss_local_and_top.backward()
            for param in self.model.global_network.parameters():
                param.requires_grad = True

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss_local_and_top.item()
            loss_list.append(loss_local_and_top.item())
            wandb.log({'Training loss': loss_local_and_top.item(),
                       'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)


class TCACNetTrainer(Trainer):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        super(TCACNetTrainer, self).__init__(args, local_rank=local_rank, task_id=task_id, subject_id=subject_id)

    def prepare_inputs_kwargs(self, inputs):
        time_series = inputs['time_series']
        wpser = self.wpser(time_series)
        labels = inputs['labels']
        if self.model.training and self.args.mix_up:
            time_series, wpser, labels, _ = continues_mixup_data(
                # time_series, node_feature, y1=labels.float(), alpha=self.data_config.alpha, beta=self.data_config.beta)
                time_series, wpser, y1=labels.float())
        return {"time_series": time_series.to(self.device),
                "node_feature": wpser.to(self.device),
                "labels": labels.float().to(self.device)}

    @staticmethod
    def wpser(time_series):
        fs = 200
        lowcut = 8
        highcut = 30
        order = 4
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', output='ba')
        time_series = signal.filtfilt(b, a, time_series)
        _, n, _ = time_series.shape
        coeffs = pywt.wavedec(time_series, 'db4', level=5)
        energy = np.array([np.square(level).sum(-1) for level in coeffs])
        wpser = energy / np.repeat(np.expand_dims(energy.sum(-1), -1), n, -1)
        wpser = wpser.sum(0)
        wpser = torch.from_numpy(wpser).float()
        return wpser

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss_global_model = outputs.loss_global_model
            loss_local_and_top = outputs.loss_local_and_top

            for param in self.model.local_network.parameters():
                param.requires_grad = False
            for param in self.model.top_layer.parameters():
                param.requires_grad = False
            loss_global_model.backward(retain_graph=True)

            for param in self.model.local_network.parameters():
                param.requires_grad = True
            for param in self.model.top_layer.parameters():
                param.requires_grad = True
            for param in self.model.global_network.parameters():
                param.requires_grad = False
            loss_local_and_top.backward()
            for param in self.model.global_network.parameters():
                param.requires_grad = True

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule

            losses += loss_global_model.item()
            loss_list.append(loss_global_model.item())
            wandb.log({'Training loss': loss_global_model.item(),
                       'Training local loss': loss_local_and_top.item(),
                       'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

