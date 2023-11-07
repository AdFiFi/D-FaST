import json
import os
from timeit import default_timer as timer
import wandb
import logging
import torch
import numpy as np
from abc import abstractmethod
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, classification_report

from config import init_model_config
from .optimizer import init_optimizer
from .schedule import init_schedule
from .accuracy import accuracy
from data import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self, args, local_rank=0, task_id=0, subject_id=0):
        self.task_id = task_id
        self.args = args
        self.local_rank = local_rank
        self.subject_id = subject_id
        self.data_config = DataConfig(args)
        self.data_loaders = self.load_datasets()

        model, self.model_config = init_model_config(args, self.data_config)
        if args.do_parallel:
            # self.model = torch.nn.DataParallel(self.model)
            self.device = f'cuda:{self.local_rank}' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   find_unused_parameters=True)
        else:
            self.device = f'cuda' \
                if args.device != 'cpu' and torch.cuda.is_available() else args.device
            self.model = model.to(args.device)
        # self.model = torch.compile(model, dynamic=True)

        self.optimizer = None
        self.scheduler = None

        self.best_result = None
        self.test_result = None

    @abstractmethod
    def prepare_inputs_kwargs(self, inputs):
        return {}

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

    def init_components(self):
        total = self.args.num_epochs * len(self.data_loaders['train'])
        self.optimizer = init_optimizer(self.model, self.args)
        self.scheduler = init_schedule(self.optimizer, self.args, total)

    def train_epoch(self):
        train_dataloader = self.data_loaders['train']
        self.model.train()
        losses = 0
        loss_list = []

        for step, inputs in enumerate(train_dataloader):
            # with torch.autograd.set_detect_anomaly(True):
            input_kwargs = self.prepare_inputs_kwargs(inputs)
            outputs = self.model(**input_kwargs)
            loss = outputs.loss

            if self.data_config.dataset == "ZuCo":
                loss.backward()
                if step % self.data_config.batch_size == self.data_config.batch_size - 1:

                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule

            losses += loss.item()
            loss_list.append(loss.item())
            wandb.log({'Training loss': loss.item(),
                       'Learning rate': self.optimizer.param_groups[0]['lr']})

        return losses / len(loss_list)

    def train(self):
        total = self.args.num_epochs*len(self.data_loaders['train'])
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.data_loaders['train']))
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  warmup steps = %d", self.args.warmup_steps)
        logger.info("  Total optimization steps = %d", total)
        logger.info("  Save steps = %d", self.args.save_steps)

        self.init_components()
        if self.args.visualize:
            self.visualize()
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="epoch", ncols=0):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            self.data_config.alpha = self.data_config.beta = \
                0.5 * (self.args.num_epochs - epoch) / self.args.num_epochs + 0.5
            self.test_result = self.evaluate()
            msg = f" Train loss: {train_loss:.5f}, Test loss: {self.test_result['Loss']:.5f}," \
                  f"Epoch time = {(end_time - start_time):.3f}s"
            print(msg)
            logger.info(msg)
            if self.best_result is None or self.best_result['Accuracy'] <= self.test_result['Accuracy']:
                self.best_result = self.test_result
                self.save_model()
        wandb.log({f"Best {k}": v for k, v in self.best_result.items()})

    def evaluate(self):
        if self.data_config.num_class == 2:
            result = self.binary_evaluate()
        else:
            result = self.multiple_evaluate()
        return result

    def binary_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = []
        acc = []
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")

                top1 = accuracy(outputs.logits, input_kwargs['labels'][:, 1])[0]
                acc.append([top1*input_kwargs['labels'].shape[0], input_kwargs['labels'].shape[0]])
                preds += F.softmax(outputs.logits, dim=1)[:, 1].tolist()
                labels += input_kwargs['labels'][:, 1].tolist()
            acc = np.array(acc).sum(axis=0)
            result['Accuracy'] = acc[0] / acc[1]
            result['AUC'] = roc_auc_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0
            metric = precision_recall_fscore_support(
                labels, preds, average='micro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            report = classification_report(
                labels, preds, output_dict=True, zero_division=0)

            recall = [0, 0]
            for k, v in report.items():
                if '.' in k:
                    recall[int(float(k))] = v['recall']

            result['Specificity'] = recall[0]
            result['Sensitivity'] = recall[1]
            result['Loss'] = losses / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        else:
            print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}, '
                  f'Specificity:{result["Specificity"]:.5f}, Sensitivity:{result["Sensitivity"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(result)
        return result

    def multiple_evaluate(self):
        logger.info(f"***** Running evaluation on test{self.task_id} dataset *****")
        self.model.eval()
        evaluate_dataloader = self.data_loaders['test']
        losses = 0
        loss_list = []
        labels = []
        result = {}
        preds = None
        with torch.no_grad():
            for inputs in evaluate_dataloader:
                input_kwargs = self.prepare_inputs_kwargs(inputs)
                outputs = self.model(**input_kwargs)
                loss = outputs.loss
                losses += loss.item()
                loss_list.append(loss.item())
                # print(f"Evaluate loss: {loss.item():.5f}")
                if preds is None:
                    preds = F.softmax(outputs.logits, dim=1).cpu().numpy()
                else:
                    preds = np.append(preds, F.softmax(outputs.logits, dim=1).cpu().numpy(), axis=0)
                labels += input_kwargs['labels'].argmax(dim=-1).tolist()
            try:
                result['AUC'] = roc_auc_score(labels, preds, multi_class='ovo')
            except:
                result['AUC'] = 0.
            preds = preds.argmax(axis=1).tolist()
            result['Accuracy'] = accuracy_score(labels, preds)
            preds, labels = np.array(preds), np.array(labels)
            metric = precision_recall_fscore_support(
                labels, preds, average='micro')
            result['Precision'] = metric[0]
            result['Recall'] = metric[1]
            result['F_score'] = metric[2]

            # report = classification_report(
            #     labels, preds, output_dict=True, zero_division=0)
            #
            # recall = [0, 0]
            # for k, v in report.items():
            #     if '.' in k:
            #         recall[int(float(k))] = v['recall']
            #
            # result['Specificity'] = recall[0]
            # result['Sensitivity'] = recall[1]
            result['Loss'] = losses / len(loss_list)
        if self.args.within_subject:
            print(f'Test{self.subject_id}-{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        else:
            print(f'Test{self.task_id} : Accuracy:{result["Accuracy"]:.5f}, AUC:{result["AUC"]:.5f}', end=',')
        for k, v in result.items():
            if v is not None:
                logger.info(f"{k}: {v:.5f}")
        wandb.log(result)
        return result

    def save_model(self):
        # Save model checkpoint (Overwrite)
        path = os.path.join(self.args.model_dir, self.args.model)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = self.model.module if self.args.do_parallel else self.model
        torch.save(model_to_save, os.path.join(path, f'{self.args.model}-{self.task_id}.bin'))

        # Save training arguments together with the trained model
        args_dict = {k: v for k, v in self.args.__dict__.items()}
        with open(os.path.join(path, "config.json"), 'w') as f:
            f.write(json.dumps(args_dict))
        logger.info("Saving model checkpoint to %s", path)

    def load_model(self):
        path = os.path.join(self.args.model_dir, f'{self.args.model}-{self.task_id}.bin')
        if not os.path.exists(path):
            logger.info("Model doesn't exists! Train first!")
            return
            # raise Exception("Model doesn't exists! Train first!")

        if self.args.do_parallel:
            self.model = self.model.to(self.args.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])
        else:
            self.model = torch.load(os.path.join(path))
            self.model.to(self.device)
        logger.info("***** Model Loaded *****")

    def visualize(self):
        self.model.eval()
        inputs = (torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.time_series_size)),
                  torch.rand((self.data_config.batch_size, self.data_config.node_size, self.data_config.node_size)),
                  F.one_hot(torch.randint(0, self.model_config.num_classes, (self.data_config.batch_size,))))
        input_kwargs = self.prepare_inputs_kwargs(inputs)
        # save_path = os.path.join(self.args.model_dir, self.args.model, 'model.onnx')
        self.model.config.dict_output = False
        torch.onnx.export(self.model,
                          tuple([v for k, v in input_kwargs.items()]),
                          'model.onnx')
        wandb.save('model.onnx')
        self.model.config.dict_output = True
