import torch
import numpy as np
import os
import cv2

from src.data.as_dataloader import class_labels
from src.utils.utils import makedir, load_pickle, save_pickle
from tqdm import tqdm


def load_data_and_model_products(model, dataloader, mode, data_config, abstain_class, root_dir_for_saving, log=print):
    """
    to run the model with a dataset of interest for 1 epoch and extract the model products and saves them alongside
    the dataset split information and images in pickle files
    :return: loaded data_dict and model_products_dic
    """
    filename = (
        f'{data_config["view"]}_'
        f'{data_config["frames"]}x{data_config["img_size"]}_'
        f'{data_config["interval_quant"]:.1f}x{data_config["interval_unit"]}_'
        f'{"all-Intervals" if data_config["iterate_intervals"] else ""}_'
        f"{mode}_data"
    )
    data_dict_path = f'{data_config["dataset_root"]}/pickled_datasets/{filename}.pickle'
    makedir(os.path.dirname(data_dict_path))
    model_products_path = f"{root_dir_for_saving}/{mode}/model_products.pickle"
    makedir(os.path.dirname(model_products_path))
    if os.path.exists(data_dict_path) and os.path.exists(model_products_path):
        data_dict = load_pickle(data_dict_path, log)
        log(f"img  and labels and filenames of {mode}-dataset is loaded")

        model_products_dict = load_pickle(model_products_path, log)
        log(
            f"model products: prototypical layer input, similarity scores (distances), "
            f"ROI maps, predictions, and FC Layer weights for model {root_dir_for_saving} "
            f"for {mode}-dataset is loaded"
        )
    else:
        log(f"model products not saved. running the epoch on {mode}-dataset to save the results.")
        protoL_input_ = []  # saves the input to prototypical layer (conv feature * occurrence map), shape (N, P, D)
        proto_dist_ = []  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (N, P)
        occurrence_map_ = []  # saves the computed occurence maps. shape (N, P, 1, (T), H, W)
        inputs = []  # saves the input images. shape (N, P, 3, (To), Ho, Wo)
        ys_gt = []
        ys_pred = []
        filenames = []
        fc_layer_weights = model.last_layer.weight.detach().cpu().numpy()

        data_iter = iter(dataloader)
        iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
        for i in iterator:
            data_sample = next(data_iter)
            search_batch = data_sample["cine"]  # shape (B, 3, (To), Ho, Wo)
            search_y = data_sample["target_AS"]

            inputs.extend(search_batch.detach().cpu().numpy())
            ys_gt.extend(search_y.detach().cpu().numpy())
            filenames.extend(data_sample["filename"])

            with torch.no_grad():
                search_batch = search_batch.cuda()
                (
                    protoL_input_torch,
                    proto_dist_torch,
                    occurrence_map_torch,
                    logits,
                ) = model.push_forward(search_batch)
                if abstain_class:
                    # take only logits from the non-abstention class
                    y_pred_prob = logits[:, : model.num_classes - 1].softmax(dim=1).cpu()
                else:
                    y_pred_prob = logits.softmax(dim=1).cpu()

            protoL_input_.extend(protoL_input_torch.detach().cpu().numpy())
            proto_dist_.extend(proto_dist_torch.detach().cpu().numpy())
            occurrence_map_.extend(occurrence_map_torch.detach().cpu().numpy())
            ys_pred.extend(y_pred_prob.detach().cpu().numpy())

        protoL_input_ = np.asarray(protoL_input_)  # shape (N, P, D)
        proto_dist_ = np.asarray(proto_dist_)  # shape (N, P)
        occurrence_map_ = np.asarray(occurrence_map_)  # shape (N, P, 1, (T), H, W)
        inputs = np.asarray(inputs)  # shape (N, P, 3, (To), Ho, Wo)
        ys_gt = np.asarray(ys_gt)  # shape (N)
        ys_pred = np.asarray(ys_pred)  # shape (N, classes)

        ### Sanity Check
        from sklearn.metrics import (
            multilabel_confusion_matrix,
            f1_score,
            classification_report,
        )

        ys_pred_class = ys_pred.argmax(axis=1)
        label_names = class_labels
        f1_orig = f1_score(
            ys_gt,
            ys_pred_class,
            average=None,
            labels=range(len(label_names)),
            zero_division=0,
        )
        f1_mean = f1_orig.mean()
        cm = multilabel_confusion_matrix(ys_gt, ys_pred_class, labels=range(len(label_names)))
        log(f"f1 score is {f1_orig} with mean {f1_mean}")
        log(f"confusion matrix is \n{cm}")
        log(
            classification_report(
                ys_gt,
                ys_pred_class,
                labels=range(len(label_names)),
                target_names=label_names,
                zero_division=0,
            )
        )

        data_dict = {
            "inputs": inputs,
            "ys_gt": ys_gt,
            "filenames": filenames,
        }
        model_products_dict = {
            "fc_layer_weights": fc_layer_weights,
            "protoL_input_": protoL_input_,
            "proto_dist_": proto_dist_,
            "occurrence_map_": occurrence_map_,
            "ys_pred": ys_pred,
        }

        save_pickle(data_dict, data_dict_path)
        save_pickle(model_products_dict, model_products_path)

    return data_dict, model_products_dict


def get_src(src_imgs, m=0.099, std=0.171):
    """

    :param src_imgs: shape (N, 3, (To), Ho, Wo)
    :return: reshaped images
    """
    src_imgs = src_imgs * std + m
    D = len(src_imgs.shape)
    if D == 4:  # image
        src_imgs = np.transpose(src_imgs, (0, 2, 3, 1))  # shape (N, Ho, Wo, 3)
        N, Ho, Wo, Co = src_imgs.shape
        dsize = (Ho, Wo)
        upsample_mode = "bilinear"
    elif D == 5:  # video
        src_imgs = np.transpose(src_imgs, (0, 2, 3, 4, 1))  # shape (N, To, Ho, Wo, 3)
        N, To, Ho, Wo, Co = src_imgs.shape
        dsize = (To, Ho, Wo)
        upsample_mode = "trilinear"
    upsampler = torch.nn.Upsample(size=dsize, mode=upsample_mode)

    return src_imgs, upsampler


def get_normalized_upsample_occurence_maps(occurrence_maps, upsampler):
    """

    :param occurrence_maps: shape = (P, D=1, (T), H, W)
    :param upsampler: torch upsampler to change occurence map shape to input image/videos
    :return: numpy of normalized upsampled occurence maps in shape of (P, (To), Ho, Wo)
    """
    occurrence_map_tensor = torch.from_numpy(occurrence_maps).float()  # shape = (P, D=1, (T), H, W)
    # upsampler = torch.nn.Upsample(size=dsize, mode=upsample_mode)
    upsampled_occurrence_map_tensor = upsampler(occurrence_map_tensor).squeeze()  # Shape = (P, (To), Ho, Wo)
    upsampled_occurrence_maps = upsampled_occurrence_map_tensor.numpy()  # Shape = (P, (To), Ho, Wo)

    # normalize the occurrence map
    D = len(upsampled_occurrence_maps.shape)
    axis = tuple(range(1, D))
    rescaled_occurrence_maps = upsampled_occurrence_maps - np.amin(upsampled_occurrence_maps, axis=axis, keepdims=True)
    return rescaled_occurrence_maps / (np.amax(rescaled_occurrence_maps, axis=axis, keepdims=True) + 1e-7)


def get_heatmap(rescaled_occurrence_maps):
    """

    :param rescaled_occurrence_maps: shape (P, (To), Ho, Wo)
    :return: RGB heatmap that can be added as overlay on top of image/video. shape (P, (To), Ho, Wo, 3)
    """
    D = len(rescaled_occurrence_maps.shape)
    n_prototypes = rescaled_occurrence_maps.shape[0]
    if D == 3:
        prots_heatmaps = [
            cv2.applyColorMap(np.uint8(255 * rescaled_occurrence_maps[p]), cv2.COLORMAP_TURBO)
            for p in range(n_prototypes)
        ]
    elif D == 4:
        To = rescaled_occurrence_maps.shape[1]
        prots_heatmaps = []
        for p in range(n_prototypes):
            prots_heatmaps.append(
                np.asarray(
                    [
                        cv2.applyColorMap(
                            np.uint8(255 * rescaled_occurrence_maps[p, t]),
                            cv2.COLORMAP_TURBO,
                        )
                        for t in range(To)
                    ]
                )
            )
    prots_heatmaps = np.float32(prots_heatmaps) / 255
    prots_heatmaps = prots_heatmaps[..., ::-1]
    return prots_heatmaps
