##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, count_acc, compute_confidence_interval, ensure_path, count_dacc, \
    get_hardness_correct
from utils.hacc import Hacc
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
import configs
from utils.misc import pprint
import pickle
from utils.misc import progress_bar, get_top_k_losses
from utils.early_stopping import EarlyStopping

import wandb

USER_NAME = 'freefeynman123'

class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""

    def __init__(self, args):
        param = configs.__dict__[args.config]()
        args.shot = param.shot
        args.test = param.test
        args.debug = param.debug
        args.deconfound = param.deconfound
        args.meta_label = param.meta_label
        args.init_weights = param.init_weights
        self.test_iter = param.test_iter
        args.param = param
        pprint(vars(args))

        # Set the folder to save the records and checkpoints
        log_base_dir = '/nas/people/lukasz_bala/reproducibility/ifsl/MTL/logs/baseline'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
                     '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr1' + str(
            args.meta_lr1) + '_lr2' + str(args.meta_lr2) + \
                     '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
                     '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
                     '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        self.trainset = Dataset('train', self.args, dataset=self.args.param.dataset, train_aug=False)
        num_workers = self.args.num_workers
        if args.debug:
            num_workers = 0
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way,
                                               self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=num_workers,
                                       pin_memory=True)

        # Load meta-val set
        self.valset = Dataset('val', self.args, dataset=self.args.param.dataset, train_aug=False,
                              require_index=self.args.require_index)
        self.val_sampler = CategoriesSampler(self.valset.label, self.test_iter, self.args.way,
                                             self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=num_workers,
                                     pin_memory=True)

        # Build meta-transfer learning model
        self.model = MtlLearner(self.args)
        if self.args.wandb_id is None:
            id = wandb.util.generate_id()
            # load pretrained model without FC classifier
            self.model.load_pretrain_weight(self.args.init_weights)
        else:
            id = self.args.wandb_id
        if self.args.phase == 'meta_train':
            self.run_train = wandb.init(project=self.args.project_name, resume="allow", job_type='train', entity=USER_NAME)
        else:
            self.run_eval = wandb.init(project=self.args.project_name, resume="allow", job_type='eval', entity=USER_NAME)
        if self.args.wandb_id is not None:
            checkpoint = torch.load(wandb.restore("max_acc.pth").name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        wandb.config.update(self.args)

        '''
        self.model_dict = self.model.state_dict()
        if self.args.init_weights is not None:
            pretrained_dict = torch.load(self.args.init_weights)['params']
        else:
            pre_base_dir = osp.join(log_base_dir, 'pre')
            pre_save_path1 = '_'.join([args.dataset, args.model_type])
            pre_save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
                str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
            pre_save_path = pre_base_dir + '/' + pre_save_path1 + '_' + pre_save_path2
            pretrained_dict = torch.load(osp.join(pre_save_path, 'max_acc.pth'))['params']
        pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
        print(pretrained_dict.keys())
        self.model_dict.update(pretrained_dict)
        self.model.load_state_dict(self.model_dict)
        '''

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            if self.args.param.model == "wideres":
                print("Using Parallel")
                self.model.encoder = torch.nn.DataParallel(self.model.encoder).cuda()

        # Set optimizer
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())},
             {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr2}], lr=self.args.meta_lr1)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                            gamma=self.args.gamma)

        if not self.args.deconfound:
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = torch.nn.NLLLoss().cuda()

        # Enable evaluation with Cross
        if args.cross:
            args.param.dataset = "cross"

    def write_output_message(self, message, file_name=None):
        if file_name is None:
            file_name = "results"
        # output_file = os.path.join(self.args.save_path, "results.txt")
        output_file = os.path.join("outputs", file_name + ".txt")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as f:
            f.write(message + "\n")

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.h5'))

    def save_artifact(self, name):
        """Saves artifact for given model name
        Args:
            name: the name for saved checkpoint
        """
        artifact = wandb.Artifact(type='model', name=''.join([self.args.model_type, self.args.config]))
        artifact.add_file(osp.join(self.args.save_path, name + '.h5'))
        if self.args.phase == 'meta_train':
            self.run_train.log_artifact(artifact)
        elif self.args.phase == 'meta_eval':
            self.run_eval.log_artifact(artifact)

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
        #Initializing early stopping class
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Generate the labels for test set of the episodes during meta-train updates
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                # Output logits for model
                logits = self.model((data_shot, label_shot, data_query, False))
                # Calculate meta-train loss
                loss = self.criterion(logits, label)
                # Calculate meta-train accuracy
                predictions = F.softmax(logits, dim=1).argmax(dim=1)
                acc = count_acc(predictions, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            # Aggregate labels, losses and predictions for logging

            val_losses, val_labels, val_predictions, val_indices, val_label_names = [], [], [], [], []

            # Generate the labels for test set of the episodes during meta-val for this epoch
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            print_freq = int(self.test_iter / 5)

            if epoch > 0:
                for i, batch in enumerate(self.val_loader, 1):
                    if len(batch) == 4:
                        if torch.cuda.is_available():
                            data = batch[0].cuda()
                        else:
                            data = batch[0]
                        indices = batch[2]
                        label_names = batch[3]
                    else:
                        if torch.cuda.is_available():
                            data = batch[0].cuda()
                        else:
                            data = batch[0]
                    p = self.args.shot * self.args.way
                    data_shot, data_query = data[:p], data[p:]
                    logits = self.model((data_shot, label_shot, data_query, True))
                    # loss = F.cross_entropy(logits, label)
                    if not self.args.deconfound:
                        loss = F.cross_entropy(logits, label)
                        loss_for_logs = F.cross_entropy(logits, label, reduction='none')
                    else:
                        loss = F.nll_loss(logits, label)
                        loss_for_logs = F.nll_loss(logits, label, reduction='none')

                    predictions = F.softmax(logits, dim=1).argmax(dim=1)
                    acc = count_acc(predictions, label)

                    val_loss_averager.add(loss.item())
                    val_acc_averager.add(acc)
                    val_losses.append(loss_for_logs)
                    val_predictions.append(predictions)
                    val_labels.append(label)
                    val_indices.append(indices[p:])
                    val_label_names.append(label_names[p:])

                    if i % print_freq == 0:
                        # Update validation averagers
                        val_loss_averager_item = val_loss_averager.item()
                        val_acc = val_acc_averager.item()
                        # Write the tensorboardX records
                        writer.add_scalar('data/val_loss', float(val_loss_averager_item), epoch)
                        writer.add_scalar('data/val_acc', float(val_acc), epoch)
                        # Print loss and accuracy for this epoch
                        print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager_item,
                                                                             val_acc))

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc = val_acc_averager.item()
            data_k, losses_k, labels_k, predictions_k, label_names_k, prediction_names_k = get_top_k_losses(
                self.valset.data, val_losses, val_labels,
                val_predictions,
                val_indices,
                val_label_names)
            images_to_log = [
                wandb.Image(data, caption=f"Image with label name: {label_name} prediction name: {prediction_name}") for
                data, loss, label_name, prediction_name in
                zip(data_k, losses_k, label_names_k, prediction_names_k)]
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_loss', float(val_loss_averager_item), epoch)
            writer.add_scalar('data/val_acc', float(val_acc), epoch)
            # Print loss and accuracy for this epoch
            msg = 'Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc)
            print(msg)
            self.write_output_message(msg)

            # Update best saved model
            if val_acc > trlog['max_acc']:
                trlog['max_acc'] = val_acc
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
                self.save_artifact('max_acc')
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch' + str(epoch))
                self.save_artifact('epoch' + str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc)

            wandb.log(
                {'train_loss': train_loss_averager, 'train_acc': train_acc_averager,
                 'val_loss': val_loss_averager, 'val_acc': val_acc, "examples": images_to_log})

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        early_stopping(val_loss_averager, self.model)
        writer.close()

    def eval(self):
        """The function for the meta-eval phase."""
        # Load the logs
        # trlog = torch.load(osp.join(self.args.save_path, 'trlog'))

        num_workers = self.args.num_workers
        if self.args.debug:
            num_workers = 0

        self.test_iter = 2000
        # Load meta-test set
        test_set = Dataset('test', self.args, dataset=self.args.param.dataset, train_aug=False)
        sampler = CategoriesSampler(test_set.label, self.test_iter, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)

        # Set test accuracy recorder
        test_acc_record = np.zeros((self.test_iter,))

        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            # Load according to config file
            args = self.args
            base_path = "/nas/people/lukasz_bala/reproducibility/ifsl/MTL/logs/ifsl"
            if args.param.dataset == "tiered":
                add_path = "tiered_"
            else:
                add_path = ""
            if args.param.model == "ResNet10":
                add_path += "resnet_"
            elif args.param.model == "wideres":
                add_path += "wrn_"
            elif "baseline" in args.config:
                add_path += "baseline_"
            else:
                add_path += "edsplit_"
            add_path += str(args.param.shot)
            self.add_path = add_path
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.h5'))['params'])
        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()
        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        hacc = Hacc()
        # Start meta-test
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]
            k = self.args.way * self.args.shot
            data_shot, data_query = data[:k], data[k:]
            data_shot, data_query = data[:k], data[k:]
            logits = self.model((data_shot, label_shot, data_query, True))
            predictions = F.softmax(logits, dim=1).argmax(dim=1)
            acc = count_acc(predictions, label)
            hardness, correct = get_hardness_correct(logits, label_shot, label, data_shot, data_query,
                                                     self.model.pretrain)
            ave_acc.add(acc)
            hacc.add_data(hardness, correct)
            test_acc_record[i - 1] = acc
            if i % 100 == 0:
                # print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
                print("Average acc:{:.4f}, Average hAcc:{:.4f}".format(ave_acc.item(), hacc.get_topk_hard_acc()))

        # Modify add path to generate test case name:
        test_case_name = self.add_path
        if self.args.cross:
            test_case_name += "_cross"
        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        msg = test_case_name + ' Test Acc {:.4f} +- {:.4f}, hAcc {:.4f}'.format(ave_acc.item() * 100, pm * 100,
                                                                                hacc.get_topk_hard_acc())
        print(msg)
        self.write_output_message(msg, test_case_name)

        if self.args.save_hacc:
            print("Saving hacc!")
            pickle.dump(hacc, open("hacc/" + test_case_name, "wb"))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
