"""
Agent for video-based XprotoNet network, which is also used for our model ProtoASNet
trained end-to-end, inherits the image-based agent.
"""
import os
import numpy as np
import pandas as pd
import time
import wandb
import logging

import torch
from torch.backends import cudnn
from torchsummary import summary

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

from src.agents.XProtoNet_e2e import XProtoNet_e2e
from src.data.as_dataloader import class_labels
from src.utils.utils import makedir

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way


class Video_XProtoNet_e2e(XProtoNet_e2e):
    def __init__(self, config):
        super().__init__(config)

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
                occurrence_map_lnorm = self.Lnorm_occurrence.compute(occurrence_map, dim=(-3, -2, -1))
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
                cm = confusion_matrix(y_true, y_pred_class, labels=range(len(label_names)))
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
            logging.exception("AUC calculation failed, setting it to 0")
            AUC = 0

        total_loss /= n_batches

        cm = confusion_matrix(y_true_all, y_pred_class_all, labels=range(len(label_names)))

        # Diversity Metric Calculations
        # count how many prototypes were activated in at least 1% of the samples
        div_threshold = 0.05
        diversity = np.sum(count_array[:30] > div_threshold * len(y_true_all))
        diversity_log = f"diversity: {diversity}"
        if self.config["abstain_class"]:
            diversity_abstain = np.sum(count_array[30:] > div_threshold * len(y_true_all))
            diversity_log += f" | diversity_abstain: {diversity_abstain}"
        sorted_simscore_cumsum, sorted_indices = torch.sort(simscore_cumsum, descending=True)
        logging.info(f"sorted_simscore_cumsum is {sorted_simscore_cumsum}")
        # list(zip(range(40), (count_array > 0.3 * len(y_true_all)),count_array))
        # counts, bin_edges = np.histogram(count_array)
        # import termplotlib as tpl
        # # fig = tpl.figure()
        # # fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
        # # fig.show()
        # fig = tpl.figure()
        # x = np.arange(0, len(count_array))
        # fig.plot(x, count_array)
        # fig.show()

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

        # ########################## Logging epoch information on Wandb ###############################
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

    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        frames = self.data_config["frames"]
        # summary(self.model, torch.rand((self.train_config['batch_size'], 3, img_size, img_size)))
        summary(self.model, (3, frames, img_size, img_size), device="cpu")
        # print(self.model)
