"""
Agent for the image-based XprotoNet network, trained end-to-end, inherits the non-end-to-end agent.
"""
import os

import torch
import torch.optim as optim
from torch.backends import cudnn
import logging

from copy import deepcopy

from src.agents.XProtoNet_Base import XProtoNet_Base

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way


class XProtoNet_e2e(XProtoNet_Base):
    def __init__(self, config):
        super().__init__(config)

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])
        optimizer_name = config.pop("name")
        optimizer_mode = config.pop("mode")
        if optimizer_mode == "lr_same":
            optimizer_specs = [
                {
                    "params": self.model.parameters(),
                    "lr": config["lr_same"],
                    "weight_decay": 1e-3,
                }
            ]
        elif optimizer_mode == "lr_disjoint":
            optimizer_specs = [
                {
                    "params": self.model.cnn_backbone.parameters(),
                    "lr": config["lr_disjoint"]["cnn_backbone"],
                    "weight_decay": 1e-3,
                },
                # bias are now also being regularized
                {
                    "params": self.model.add_on_layers.parameters(),
                    "lr": config["lr_disjoint"]["add_on_layers"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": self.model.occurrence_module.parameters(),
                    "lr": config["lr_disjoint"]["occurrence_module"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": self.model.prototype_vectors,
                    "lr": config["lr_disjoint"]["prototype_vectors"],
                },
                {
                    "params": self.model.last_layer.parameters(),
                    "lr": config["lr_disjoint"]["last_layer"],
                },
            ]
        else:
            raise f"optimizer mode {optimizer_mode} not valid."

        self.optimizer = optim.__dict__[optimizer_name](optimizer_specs)  # Adam

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")
        scheduler = optim.lr_scheduler.__dict__[scheduler_name](self.optimizer, **config)
        return scheduler

    def get_state(self):
        return {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            if (file_name is not None) and (os.path.exists(file_name)):
                checkpoint = torch.load(file_name)

                self.current_epoch = checkpoint["epoch"]
                self.current_iteration = checkpoint["iteration"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info(
                    (
                        "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                            file_name, checkpoint["epoch"], checkpoint["iteration"]
                        )
                    )
                )
        except OSError as e:
            logging.info((f"Error {e}"))
            logging.info("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.info("**First time to train**")

    def train(self):
        """
        Main training loop
        """
        for epoch in range(self.current_epoch, self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            accu, mean_f1, auc = self.run_epoch(epoch, self.optimizer, mode="train")
            accu, mean_f1, auc = self.run_epoch(epoch, mode="val")
            # self.save_model_w_condition(model_dir=self.config['save_dir'], model_name= f'{epoch}nopush',
            #                             metric_dict={'f1': mean_f1},
            #                             threshold=0.65)

            if self.train_config["lr_schedule"]["name"] == "StepLR":
                self.scheduler.step()
            else:
                self.scheduler.step(mean_f1)

            if epoch == self.train_config["num_warm_epochs"]:
                self.push(replace_prototypes=False)

            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                self.push()
                accu, mean_f1, auc = self.run_epoch(epoch, mode="val_push")
                self.save_model_w_condition(
                    model_dir=self.config["save_dir"],
                    model_name=f"{epoch}push",
                    metric_dict={"f1": mean_f1},
                    threshold=0.65,  # TODO adjust based on your project
                )

                # saving best model after pushing
                is_best = mean_f1 > self.best_metric
                if is_best:
                    self.best_metric = mean_f1
                    logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                self.save_checkpoint(is_best=is_best)

            # saving last model
            self.save_checkpoint(is_best=False)

    def train_push_end(self):
        """
        Main training loop that projects only at the end of the training
        """
        for epoch in range(self.current_epoch, self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            accu, mean_f1, auc = self.run_epoch(epoch, self.optimizer, mode="train")
            accu, mean_f1, auc = self.run_epoch(epoch, mode="val")
            self.save_model_w_condition(
                model_dir=self.config["save_dir"],
                model_name=f"{epoch}nopush",
                metric_dict={"f1": mean_f1},
                threshold=0.65,  # TODO adjust based on your project
            )
            # saving best model
            is_best = mean_f1 > self.best_metric
            if is_best:
                self.best_metric = mean_f1
                logging.info(f"achieved best model with mean_f1 of {mean_f1}")
            self.save_checkpoint(is_best=is_best)

            if self.train_config["lr_schedule"]["name"] == "StepLR":
                self.scheduler.step()
            else:
                self.scheduler.step(mean_f1)

            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                self.push(replace_prototypes=False)

            # saving last model
            self.save_checkpoint(is_best=False)

        # Push at the end using the last model
        logging.info("Projecting the prototypes of the last model")
        self.push()
        accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode="val")
        self.save_model_w_condition(
            model_dir=self.config["save_dir"],
            model_name=f"{self.current_epoch}push_last",
            metric_dict={"mean_f1": mean_f1},
            threshold=0.5,  # TODO adjust based on your project
        )

        # Push at the end using the best model
        logging.info("Projecting the prototypes of the best model")
        self.load_checkpoint(os.path.join(self.config["save_dir"], f"model_best.pth"))
        self.push()
        accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode="val")
        self.save_model_w_condition(
            model_dir=self.config["save_dir"],
            model_name=f"{self.current_epoch}push_best",
            metric_dict={"mean_f1": mean_f1},
            threshold=0.5,  # TODO adjust based on your project
        )

    def log_lr(self, epoch_log_dict):
        epoch_log_dict.update({"lr": self.optimizer.param_groups[0]["lr"]})
