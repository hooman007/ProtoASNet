"""
XprotoNet agent, but trained in stages like protoPNet.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb
import logging

import torch
import torch.optim as optim
from torch.backends import cudnn

from copy import deepcopy
from tqdm import tqdm

from src.agents.ProtoPNet_Base import ProtoPNet_Base
from ..utils.metrics import SparsityMetric
from ..utils.utils import makedir
from src.loss.loss import (
    CeLoss,
    ClusterRoiFeat,
    SeparationRoiFeat,
    OrthogonalityLoss,
    L_norm,
    TransformLoss,
    CeLossAbstain,
)
from src.utils import push_abs_revision, local_explainability, global_explainability
from src.data.as_dataloader import class_labels

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way


class XProtoNet_Base(ProtoPNet_Base):
    def __init__(self, config):
        super().__init__(config)
        # Initialize metrics to quantify sparsity as percentage weights needed for explanation
        self.val_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.val_push_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.test_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.train_sparsity_80 = SparsityMetric(level=0.8, device=self.device)

    def get_criterion(self):
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        config = deepcopy(self.train_config["criterion"])

        # classification cost
        if self.config["abstain_class"]:
            self.CeLoss = CeLossAbstain(**config["CeLossAbstain"])
        else:
            self.CeLoss = CeLoss(**config["CeLoss"])

        # prototypical layer cost
        num_classes = self.model.num_classes
        self.Cluster = ClusterRoiFeat(num_classes=num_classes, **config["ClusterRoiFeat"])
        self.Separation = SeparationRoiFeat(
            num_classes=num_classes,
            **config["SeparationRoiFeat"],
            abstain_class=self.config["abstain_class"],
        )
        self.Orthogonality = OrthogonalityLoss(num_classes=num_classes, **config["OrthogonalityLoss"])

        # occurrence map regularization
        self.Lnorm_occurrence = L_norm(**config["Lnorm_occurrence"])
        self.Trans_occurrence = TransformLoss(**config["trans_occurrence"])

        # classification layer regularization
        self.Lnorm_fc = L_norm(**config["Lnorm_FC"], mask=1 - torch.t(self.model.prototype_class_identity))

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])

        joint_optimizer_specs = [
            {
                "params": self.model.cnn_backbone.parameters(),
                "lr": config["joint_lrs"]["cnn_backbone"],
                "weight_decay": 1e-3,
            },
            # bias are now also being regularized
            {
                "params": self.model.add_on_layers.parameters(),
                "lr": config["joint_lrs"]["add_on_layers"],
                "weight_decay": 1e-3,
            },
            {
                "params": self.model.occurrence_module.parameters(),
                "lr": config["joint_lrs"]["occurrence_module"],
                "weight_decay": 1e-3,
            },
            {
                "params": self.model.prototype_vectors,
                "lr": config["joint_lrs"]["prototype_vectors"],
            },
        ]
        self.joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

        warm_optimizer_specs = [
            {
                "params": self.model.add_on_layers.parameters(),
                "lr": config["warm_lrs"]["add_on_layers"],
                "weight_decay": 1e-3,
            },
            {
                "params": self.model.occurrence_module.parameters(),
                "lr": config["joint_lrs"]["occurrence_module"],
                "weight_decay": 1e-3,
            },
            {
                "params": self.model.prototype_vectors,
                "lr": config["warm_lrs"]["prototype_vectors"],
            },
        ]
        self.warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        last_layer_optimizer_specs = [
            {
                "params": self.model.last_layer.parameters(),
                "lr": config["last_layer_lr"],
            }
        ]
        self.last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")

        scheduler = {
            "joint": optim.lr_scheduler.__dict__[scheduler_name](self.joint_optimizer, **config),
            "last": optim.lr_scheduler.__dict__[scheduler_name](self.last_layer_optimizer, **config),
        }
        return scheduler

    def push(self, replace_prototypes=True):
        """
        pushing prototypes
        :param replace_prototypes: to replace prototypes with the closest features or not
        """
        epoch = f"{self.current_epoch}_pushed"
        push_abs_revision.push_prototypes(
            dataloader=self.data_loaders["train_push"],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            abstain_class=self.config["abstain_class"],
            preprocess_input_function=None,  # normalize if needed
            root_dir_for_saving_prototypes=os.path.join(self.config["save_dir"], "img"),
            # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix="prototype-img",
            prototype_self_act_filename_prefix="prototype-self-act",
            proto_bound_boxes_filename_prefix="bb",
            replace_prototypes=replace_prototypes,
        )

    def train(self):
        """
        Main training loop
        :return:
        """
        f1_prev = 0
        counter = 0
        for epoch in range(self.current_epoch, self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            # Step1: warmup: train all except CNN backbone and classification layer
            if epoch < self.train_config["num_warm_epochs"]:
                self.warm_only()
                accu, mean_f1, auc = self.run_epoch(epoch, self.warm_optimizer, mode="train")

            # Step2: train all
            else:
                self.joint()
                accu, mean_f1, auc = self.run_epoch(epoch, self.joint_optimizer, mode="train")
                if self.train_config["lr_schedule"]["name"] == "StepLR":
                    self.scheduler.step()

            if epoch == self.train_config["num_warm_epochs"]:
                self.push(replace_prototypes=False)

            accu, mean_f1, auc = self.run_epoch(epoch, mode="val")
            self.save_model_w_condition(
                model_dir=self.config["save_dir"],
                model_name=f"{epoch}nopush",
                metric_dict={"f1": mean_f1},
                threshold=0.75,  # TODO adjust based on your project
            )

            if epoch > self.train_config["num_warm_epochs"]:
                # LR scheduler step
                if self.train_config["lr_schedule"]["name"] != "StepLR":
                    self.scheduler["joint"].step(mean_f1)

                # Check for mean_f1 score stopping improvement for 3 epochs
                if mean_f1 > f1_prev:
                    counter = 0
                    f1_prev = mean_f1
                else:
                    counter += 1

            # if (counter == 3):
            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                f1_prev = mean_f1
                counter = 0

                # Step3: push prototypes
                self.push()
                accu, mean_f1, auc = self.run_epoch(epoch, mode="val_push")
                self.save_model_w_condition(
                    model_dir=self.config["save_dir"],
                    model_name=f"{epoch}push",
                    metric_dict={"f1": mean_f1},
                    threshold=0.65,  # TODO adjust based on your project
                )

                # Step4: train classification layer only
                # if self.model_config['prototype_activation_function'] != 'linear':
                self.last_only()
                for i in range(5):
                    logging.info("iteration: \t{0}".format(i))
                    accu, mean_f1, auc = self.run_epoch(epoch, self.last_layer_optimizer, mode="train")
                    accu, mean_f1, auc = self.run_epoch(epoch, mode="val_push")
                    self.save_model_w_condition(
                        model_dir=self.config["save_dir"],
                        model_name=f"{epoch}_{i}push",
                        metric_dict={"f1": mean_f1},
                        threshold=0.70,  # TODO adjust based on your project
                    )
                    self.scheduler["last"].step(mean_f1)

                    # saving best model after 4-step training
                    is_best = mean_f1 > self.best_metric
                    if is_best:
                        self.best_metric = mean_f1
                        logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                    self.save_checkpoint(is_best=is_best)
            # saving last model
            self.save_checkpoint(is_best=False)

    def warm_only(self):
        logging.info("\t#####################################################################")
        logging.info("\twarm")
        logging.info("\t#####################################################################")
        for p in self.model.cnn_backbone.parameters():
            p.requires_grad = False
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = True
        for p in self.model.occurrence_module.parameters():
            p.requires_grad = True
        self.model.prototype_vectors.requires_grad = True
        for p in self.model.last_layer.parameters():
            p.requires_grad = False

    def joint(self):
        logging.info("\t#####################################################################")
        logging.info("\tjoint")
        logging.info("\t#####################################################################")
        for p in self.model.cnn_backbone.parameters():
            p.requires_grad = True
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = True
        for p in self.model.occurrence_module.parameters():
            p.requires_grad = True
        self.model.prototype_vectors.requires_grad = True
        for p in self.model.last_layer.parameters():
            p.requires_grad = False

    def last_only(self):
        logging.info("\t#######################")
        logging.info("\tlast layer")
        logging.info("\t#######################")
        for p in self.model.cnn_backbone.parameters():
            p.requires_grad = False
        for p in self.model.add_on_layers.parameters():
            p.requires_grad = False
        for p in self.model.occurrence_module.parameters():
            p.requires_grad = False
        self.model.prototype_vectors.requires_grad = False
        for p in self.model.last_layer.parameters():
            p.requires_grad = True

    def run_epoch(self, epoch, optimizer=None, mode="train"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            # if val_push, use val for dataloder
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        label_names = class_labels
        logit_names = label_names + ["abstain"] if self.config["abstain_class"] else label_names

        n_batches = 0
        total_loss = np.zeros(7)

        y_pred_class_all = torch.FloatTensor()
        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        # Diversity Metric
        count_array = np.zeros(self.model.prototype_shape[0])
        simscore_cumsum = torch.zeros(self.model.prototype_shape[0])

        # Reset sparsity metric
        getattr(self, f"{mode}_sparsity_80").reset()

        with torch.set_grad_enabled(mode == "train"):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            accu_batch = 0
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_AS"].to(self.device)

                logit, similarities, occurrence_map = self.model(input)

                ############ Compute Loss ###############
                # CrossEntropy loss for Multiclass data
                ce_loss = self.CeLoss.compute(logits=logit, target=target)
                # cluster cost
                cluster_cost = self.Cluster.compute(similarities, target)
                # separation cost
                separation_cost = self.Separation.compute(similarities, target)
                # to encourage diversity on learned prototypes
                orthogonality_loss = self.Orthogonality.compute(self.model.prototype_vectors)
                # occurrence map L2 regularization
                occurrence_map_lnorm = self.Lnorm_occurrence.compute(occurrence_map, dim=(-2, -1))
                # occurrence map transformation regularization
                occurrence_map_trans = self.Trans_occurrence.compute(input, occurrence_map, self.model)
                # FC layer L1 regularization
                fc_lnorm = self.Lnorm_fc.compute(self.model.last_layer.weight)

                loss = (
                    ce_loss
                    + cluster_cost
                    + separation_cost
                    + orthogonality_loss
                    + occurrence_map_lnorm
                    + occurrence_map_trans
                    + fc_lnorm
                )

                ####### evaluation statistics ##########
                if self.config["abstain_class"]:
                    # take only logits from the non-abstention class
                    y_pred_prob = logit[:, : self.model.num_classes - 1].softmax(dim=1).cpu()
                else:
                    y_pred_prob = logit.softmax(dim=1).cpu()
                y_pred_max_prob, y_pred_class = y_pred_prob.max(dim=1)
                y_pred_class_all = torch.concat([y_pred_class_all, y_pred_class])
                y_pred_all = torch.concat([y_pred_all, y_pred_prob.detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])

                # f1 score
                f1_batch = f1_score(
                    y_true.numpy(),
                    y_pred_class.numpy(),
                    average=None,
                    labels=range(len(label_names)),
                    zero_division=0,
                )
                # confusion matrix
                # cm = confusion_matrix(y_true, y_pred_class, labels=range(len(label_names)))
                # Accuracy
                accu_batch = balanced_accuracy_score(y_true.numpy(), y_pred_class.numpy())

                if mode == "train":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    self.current_iteration += 1

                total_loss += np.asarray(
                    [
                        ce_loss.item(),
                        cluster_cost.item(),
                        separation_cost.item(),
                        orthogonality_loss.item(),  # prototypical layer
                        occurrence_map_lnorm.item(),
                        occurrence_map_trans.item(),  # ROI layer
                        fc_lnorm.item(),  # FC layer
                    ]
                )
                n_batches += 1

                sparsity_batch = getattr(self, f"{mode}_sparsity_80")(similarities).item()

                # Determine the top 5 most similar prototypes to data
                # sort similarities in descending order
                sorted_similarities, sorted_indices = torch.sort(similarities[:, :30].detach().cpu(), descending=True)
                # Add the type 5 most similar prototypes to the count array
                np.add.at(count_array[:30], sorted_indices[:, :5], 1)

                if self.config["abstain_class"]:
                    # sort similarities in descending order
                    sorted_similarities, sorted_indices = torch.sort(
                        similarities[:, 30:].detach().cpu(), descending=True
                    )
                    # Add the type 5 most similar prototypes to the count array
                    np.add.at(count_array[30:], sorted_indices[:, :2], 1)

                simscore_cumsum += similarities.sum(dim=0).detach().cpu()

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"CE loss {ce_loss.item():.2f} | "
                    f"Cls {cluster_cost.item():.2f} | "
                    f"Sep {separation_cost.item():.2f} | "
                    f"Ortho {orthogonality_loss.item():.2f} | "
                    f"om_l2 {occurrence_map_lnorm.item():.4f} | "
                    f"om_trns {occurrence_map_trans.item():.2f} | "
                    f"fc_l1 {fc_lnorm.item():.4f} | "
                    f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |"
                    f"Sparsity: {sparsity_batch:.1f}",
                    refresh=True,
                )

                # ########################## Logging batch information on Wandb ###############################
                if self.config["wandb_mode"] != "disabled":
                    batch_log_dict.update(
                        {
                            # mode is 'val', 'val_push', or 'train
                            f"batch_{mode}/step": step,
                            # ######################## Loss Values #######################
                            f"batch_{mode}/loss_all": loss.item(),
                            # f'batch_{mode}/loss_Fl': focal_loss.item(),
                            f"batch_{mode}/loss_CE": ce_loss,
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_Sep": separation_cost.item(),
                            f"batch_{mode}/loss_Ortho": orthogonality_loss.item(),
                            f"batch_{mode}/loss_RoiNorm": occurrence_map_lnorm.item(),
                            f"batch_{mode}/loss_RoiTrans": occurrence_map_trans.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                            f"batch_{mode}/sparsity": sparsity_batch,
                        }
                    )
                    batch_log_dict.update(
                        {f"batch_{mode}/f1_{as_label}": value for as_label, value in zip(label_names, f1_batch)}
                    )
                    # logging all information
                    wandb.log(batch_log_dict)

                # save model preds in CSV
                if mode == "val_push" or mode == "test":
                    # ##### creating the prediction log table for saving the performance for each case
                    epoch_pred_log_df = pd.concat(
                        [
                            epoch_pred_log_df,
                            self.create_pred_log_df(
                                data_sample,
                                logit.detach().cpu(),
                                logit_names=logit_names,
                            ),
                        ],
                        axis=0,
                    )

        end = time.time()

        ######################################################################################
        # ###################################### Calculating Metrics #########################
        ######################################################################################
        y_pred_class_all = y_pred_class_all.numpy()
        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()

        accu = balanced_accuracy_score(y_true_all, y_pred_class_all)
        f1 = f1_score(
            y_true_all,
            y_pred_class_all,
            average=None,
            labels=range(len(label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()

        # AUC = roc_auc_score(y_true_all, y_pred_all, average='weighted', multi_class='ovr',
        #                     labels=range(len(label_names)))
        try:
            AUC = roc_auc_score(
                y_true_all,
                y_pred_all,
                average="weighted",
                multi_class="ovr",
                labels=range(len(label_names)),
            )
        except ValueError:
            logging.info("AUC calculation failed, setting it to 0")
            AUC = 0

        total_loss /= n_batches

        cm = confusion_matrix(y_true_all, y_pred_class_all, labels=range(len(label_names)))

        # Diversity Metric Calculations
        # count how many prototypes were activated in at least 1% of the samples
        diversity = np.sum(count_array[:30] > 0.3 * len(y_true_all))
        diversity_log = f"diversity: {diversity}"
        if self.config["abstain_class"]:
            diversity_abstain = np.sum(count_array[30:] > 0.3 * len(y_true_all))
            diversity_log += f" | diversity_abstain: {diversity_abstain}"
        sorted_simscore_cumsum, sorted_indices = torch.sort(simscore_cumsum, descending=True)
        logging.info(f"sorted_simscore_cumsum is {sorted_simscore_cumsum}")

        sparsity_epoch = getattr(self, f"{mode}_sparsity_80").compute().item()

        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "test":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_class_all}")

        logging.info(
            f"Epoch:{epoch}_{mode} | Time:{end - start:.0f} | Total_Loss:{total_loss.sum() :.3f} | "
            f"[ce, clst, sep, ortho, om_l2, om_trns, fc_l1]={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} \n"
            f"Acc: {accu:.2%} | f1: {[f'{f1[j]:.2%}' for j in range(f1.shape[0])]} | f1_avg: {f1_mean:.4f} | AUC: {AUC} \n"
            f"Sparsity: {sparsity_epoch}  |  {diversity_log}"
        )
        logging.info(f"\tConfusion matrix: \n {cm}")
        logging.info(classification_report(y_true_all, y_pred_class_all, zero_division=0, target_names=label_names))

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        if mode == "val_push" or mode == "test":
            path_to_csv = os.path.join(self.config["save_dir"], f"csv_{mode}")
            makedir(path_to_csv)
            # epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(path_to_csv, f'e{epoch:02d}_Auc{AUC.mean():.0%}.csv'))
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_f1_{f1_mean:.0%}.csv")
            )

        #################################################################################
        # ###################### Logging epoch information on Wandb #####################
        #################################################################################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss.sum(),
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1_mean": f1_mean,
                f"epoch/{mode}/accuracy": accu,
                f"epoch/{mode}/AUC_mean": AUC,
                f"epoch/{mode}/diversity": diversity,
                f"epoch/{mode}/sparsity": sparsity_epoch,
            }
            if self.config["abstain_class"]:
                epoch_log_dict.update({f"epoch/{mode}/diversity_abstain": diversity_abstain})
            self.log_lr(epoch_log_dict)
            # log f1 scores separately
            epoch_log_dict.update({f"epoch/{mode}/f1_{as_label}": value for as_label, value in zip(label_names, f1)})
            # log AUC scores separately
            # epoch_log_dict.update({
            #     f'epoch/{mode}/AUC_{as_label}': value for as_label, value in zip(label_names, AUC)
            # })
            # log losses separately
            # loss_names = ["loss_Fl", "loss_Clst", "loss_Sep", "loss_Ortho", "loss_RoiNorm", "loss_RoiTrans", "loss_fcL1Norm"]
            loss_names = [
                "loss_CE",
                "loss_Clst",
                "loss_Sep",
                "loss_Ortho",
                "loss_RoiNorm",
                "loss_RoiTrans",
                "loss_fcL1Norm",
            ]
            epoch_log_dict.update(
                {f"epoch/{mode}/{loss_name}": value for loss_name, value in zip(loss_names, total_loss)}
            )
            # logging all information
            wandb.log(epoch_log_dict)

        return accu, f1_mean, AUC

    def get_sim_scores(self, mode="train"):
        epoch = self.current_epoch

        logging.info(f"Epoch: {epoch} generating the sim scores for dataset:{mode}")
        self.model.eval()

        sim_scores = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        data_loader = self.data_loaders[mode]

        with torch.set_grad_enabled(False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                data_sample = next(data_iter)

                input = data_sample["img"].to(self.device)
                _, similarities, _ = self.model(input)
                sim_scores = torch.concat([sim_scores, similarities.cpu()])

                target = data_sample["label"].to(self.device)
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])

                iterator.set_description(f"Epoch: {epoch} | {mode} ", refresh=True)

        makedir(os.path.join(self.config["save_dir"], "ranking_prototypes"))
        torch.save(
            sim_scores,
            f=os.path.join(
                self.config["save_dir"],
                "ranking_prototypes",
                (f"sim_scores_{mode}_epoch{epoch}.pth"),
            ),
        )
        torch.save(
            y_true_all,
            f=os.path.join(self.config["save_dir"], "ranking_prototypes", (f"targets_{mode}.pth")),
        )
        # self.writer.flush()

        return

    def load_sim_scores(self, epoch, mode):
        sim_scores = torch.load(
            os.path.join(
                self.config["save_dir"],
                "ranking_prototypes",
                (f"sim_scores_{mode}_epoch{epoch}.pth"),
            )
        )
        y_true_all = torch.load(os.path.join(self.config["save_dir"], "ranking_prototypes", (f"targets_{mode}.pth")))
        return sim_scores, y_true_all

    def calc_metrics(self, logits, targets):
        label_names = class_labels
        y_true_all = targets.cpu().numpy()

        # focal_loss = self.FocalLoss.compute(pred=logits, target=targets)
        ce_loss = self.CeLoss.compute(logits=logits, target=targets)

        y_pred_all = logits.detach().cpu().numpy() > 0
        # f1 score
        f1 = f1_score(
            y_true_all,
            y_pred_all,
            average=None,
            labels=range(len(label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()
        # AUC
        AUC = roc_auc_score(y_true_all, y_pred_all, average=None, multi_class="ovr")
        AUC_mean = AUC.mean()
        # confusion matrix
        cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(label_names)))
        cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
        # Accuracy
        accu_class = []
        for j in range(y_true_all.shape[-1]):
            accu_class.append(balanced_accuracy_score(y_true_all[:, j], y_pred_all[:, j]))
        accu = np.asarray(accu_class).mean()

        return ce_loss, accu, f1_mean, AUC_mean, cm_flattened

    def explain_local(self, mode="val"):
        """
        Local explanation of caess of interest
        :param mode: dataset to select (test or val)
        """
        epoch = self.current_epoch
        local_explainability.explain_local(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            data_config=self.data_config,
            abstain_class=self.config["abstain_class"],
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
        )

    def explain_global(self, mode="val"):
        """
        Global explanation of prototypes
        :param mode: dataset to select (test or val)
        """
        pass
        # epoch = self.current_epoch
        # global_explainability.explain_global(
        #     mode=mode,  # val or test
        #     dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
        #     model=self.model,  # pytorch network with prototype_vectors
        #     preprocess_input_function=None,  # normalize if needed
        #     model_directory=self.config["save_dir"],
        #     # if not None, explainability results will be saved here
        #     epoch_number=epoch,
        # )

    def log_lr(self, epoch_log_dict):
        epoch_log_dict.update(
            {
                "lr_joint": self.joint_optimizer.param_groups[0]["lr"],
                "lr_last": self.last_layer_optimizer.param_groups[0]["lr"],
            }
        )
