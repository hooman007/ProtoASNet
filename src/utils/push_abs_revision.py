import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from src.utils.utils import makedir, save_pickle
from src.utils.video_utils import write_video, remove_images
from tqdm import tqdm


def prototype_plot_frame(
    unnorm_img,
    upsampled_occ_map,
    rescaled_occ_map,
    occ_map_min,
    occ_map_max,
    proto_id,
    fn,
    pred,
    gt,
    fig_path,
    imshow_interp_method="none",
):
    """
    Plots one frame of the prototype plot using upsampled occ_map
    Ho x Wo x 3 un-normalized [0,1] image
    Ho x Wo upsampled occurrence map
    Ho x Wo [0,1] rescaled occurrence map (normalized with respect to 3D volume if video
    """

    # image masked with normalized occurrence map
    mask = np.expand_dims(rescaled_occ_map, axis=-1)  # shape (Ho, Wo, 1)
    prototype_img = unnorm_img * mask

    # image with mask overlaid as a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_occ_map), cv2.COLORMAP_TURBO)  # shape (Ho, Wo, 3)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    overlay_img = 0.5 * unnorm_img + 0.3 * heatmap  # shape (Ho, Wo, 3)

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    images = {
        "base": unnorm_img,  # base image
        "masked": prototype_img,  # image with [0,1] mask of occurrence map
        "overlay": overlay_img,
    }  # image with colormap overlay of occurrence map

    for i, (key, v) in enumerate(images.items()):
        axs[i].imshow(v, interpolation=imshow_interp_method)
        axs[i].title.set_text(key)
    # 4. raw occurrence map (non-rescaled)
    i += 1
    im = axs[i].imshow(
        upsampled_occ_map,
        interpolation=imshow_interp_method,
        vmin=occ_map_min,
        vmax=occ_map_max,
    )
    axs[i].title.set_text("mask")
    fig.colorbar(im, ax=axs[i], shrink=0.75)

    # add super title for the figure
    fig.suptitle(
        f"p_{proto_id:02d}  | {fn} | img_pred = {[f'{pred[i]:.2f}' for i in range(pred.shape[0])]} | gt = {gt}",
        fontsize=15,
    )
    # some visual configs for the figure
    fig.tight_layout()
    plt.savefig(fig_path)

    plt.close()


def prototype_plot(
    img,
    occurrence_map,
    proto_id,
    fn,
    pred,
    gt,
    proto_dir,
    m=0.099,
    std=0.171,
    interp="none",
):
    """
    Plot a visualization pertaining to one prototype and its associated image.

    Parameters
    ----------
    img : 3 x Ho x Wo (or 3 x To x Ho x Wo for videos) ndarray
        normalized image (not in [0, 1]) where Ho and Wo denote original size
    occurrence_map : 1 x H x W ndarray for images (or 1 x T x H x W for videos)
        binary occurence mask (in [0, 1]) denoting model-predicted occurrence
    proto_id : integer
        ID number of the prototype
    fn : string
        filename of the image
    pred : K ndarray
        array indicating logits or confidences of model
    gt: int
        integer representing ground truth class
    proto_dir : os.path
        path to the save directory of prototypes
    m : float, optional
        mean to un-normalize the original image. The default is 0.099.
    std : float, optional
        stdev to un-normalize the original image. The default is 0.171.
    interp: string, optional
        interpolation method for the display purpose only. The default is 'none'.
    """

    unnorm_img = img * std + m
    D = len(unnorm_img.shape)

    if D == 3:  # image
        unnorm_img = np.transpose(unnorm_img, (1, 2, 0))  # shape (Ho, Wo, 3)
        Ho, Wo, _ = unnorm_img.shape
        dsize = (Ho, Wo)
        To = 1
        upsample_mode = "bilinear"
    elif D == 4:  # video
        unnorm_img = np.transpose(unnorm_img, (1, 2, 3, 0))  # shape (To, Ho, Wo, 3)
        To, Ho, Wo, _ = unnorm_img.shape
        dsize = (To, Ho, Wo)
        upsample_mode = "trilinear"

    # resize the occurrence map
    upsampler = torch.nn.Upsample(size=dsize, mode=upsample_mode)
    occurrence_map_tensor = torch.from_numpy(occurrence_map).float().unsqueeze(0)  # shape = (1, D=1, (T), H, W)
    upsampled_occurrence_map_tensor = upsampler(occurrence_map_tensor).squeeze()  # Shape = ((To), Ho, Wo)
    upsampled_occurrence_map = upsampled_occurrence_map_tensor.numpy()  # Shape = ((To), Ho, Wo)

    # normalize the occurrence map
    rescaled_occurrence_map = upsampled_occurrence_map - np.amin(upsampled_occurrence_map)
    rescaled_occurrence_map = rescaled_occurrence_map / (np.amax(rescaled_occurrence_map) + 1e-7)

    if D == 3:
        # plot and save the image
        fig_path = os.path.join(proto_dir, f"{proto_id:02d}_{fn}.png")
        prototype_plot_frame(
            unnorm_img,
            upsampled_occurrence_map,
            rescaled_occurrence_map,
            np.amin(upsampled_occurrence_map),
            np.amax(upsampled_occurrence_map) + 1e-7,
            proto_id,
            fn,
            pred,
            gt,
            fig_path,
            interp,
        )
    elif D == 4:
        # plot each frame, then, when all images have been plotted create a video and delete the frames
        fig_paths = []
        for t in range(To):
            fig_path = os.path.join(proto_dir, f"{proto_id:02d}_{fn}_{t}.png")
            prototype_plot_frame(
                unnorm_img[t],
                upsampled_occurrence_map[t],
                rescaled_occurrence_map[t],
                np.amin(upsampled_occurrence_map),
                np.amax(upsampled_occurrence_map) + 1e-7,
                proto_id,
                fn,
                pred,
                gt,
                fig_path,
                interp,
            )
            fig_paths.append(fig_path)
        video_path = os.path.join(proto_dir, f"{proto_id:02d}_{fn}.mp4")
        write_video(fig_paths, video_path, fps=5)
        remove_images(fig_paths)


def push_prototypes(
    dataloader,  # pytorch dataloader
    # dataset,   # pytorch dataset for train_push group
    # prototype_layer_stride=1,
    model,  # pytorch network with feature encoder and prototype vectors
    class_specific=True,  # enable pushing protos from only the alotted class
    abstain_class=True,  # indicates K+1-th class is of the "abstain" type
    preprocess_input_function=None,  # normalize if needed
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved in this dir
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    log=print,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    replace_prototypes=True,
):
    """
    Search the training set for image patches that are semantically closest to
    each learned prototype, then updates the prototypes to those image patches.

    To do this, it computes the image patch embeddings (IPBs) and saves those
    closest to the prototypes. It also saves the prototype-to-IPB distances and
    predicted occurrence maps.

    If abstain_class==True, it assumes num_classes actually equals to K+1, where
    K is the number of real classes and 1 is the extra "abstain" class for
    uncertainty estimation.
    """

    model.eval()
    log(f"############## push at epoch {epoch_number} #################")

    start = time.time()

    # creating the folder (with epoch number) to save the prototypes' info and visualizations
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    # find the number of prototypes, and number of classes for this push
    prototype_shape = model.prototype_shape  # shape (P, D, (1), 1, 1)
    P = model.num_prototypes
    proto_class_identity = np.argmax(model.prototype_class_identity.cpu().numpy(), axis=1)  # shape (P)
    proto_class_specific = np.full(P, class_specific)
    num_classes = model.num_classes
    if abstain_class:
        K = num_classes - 1
        assert K >= 2, "Abstention-push must have >= 2 classes not including abstain"
        # for the uncertainty prototypes, class_specific is False
        # for now assume that each class (inc. unc.) has P_per_class == P/num_classes
        P_per_class = P // num_classes
        proto_class_specific[K * P_per_class : P] = False
    else:
        K = num_classes

    # keep track of the input embedding closest to each prototype
    proto_dist_ = np.full(P, np.inf)  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (P)
    # save some information dynamically for each prototype
    # which are updated whenever a closer match to prototype is found
    occurrence_map_ = [None for _ in range(P)]  # saves the computed occurence maps. shape (P, 1, (T), H, W)
    # saves the input to prototypical layer (conv feature * occurrence map), shape (P, D)
    protoL_input_ = [None for _ in range(P)]
    # saves the input images with embeddings closest to each prototype. shape (P, 3, (To), Ho, Wo)
    image_ = [None for _ in range(P)]
    # saves the gt label. shape (P)
    gt_ = [None for _ in range(P)]
    # saves the prediction logits of cases seen. shape (P, K)
    pred_ = [None for _ in range(P)]
    # saves the filenames of cases closest to each prototype. shape (P)
    filename_ = [None for _ in range(P)]

    data_iter = iter(dataloader)
    iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
    for push_iter in iterator:
        data_sample = next(data_iter)

        x = data_sample["cine"]  # shape (B, 3, (To), Ho, Wo)
        if preprocess_input_function is not None:
            x = preprocess_input_function(x)

        # get the network outputs for this instance
        with torch.no_grad():
            x = x.cuda()
            (
                protoL_input_torch,
                proto_dist_torch,
                occurrence_map_torch,
                logits,
            ) = model.push_forward(x)
            pred_torch = logits.softmax(dim=1)

        # record down batch data as numpy arrays
        protoL_input = protoL_input_torch.detach().cpu().numpy()  # shape (B, P, D)
        proto_dist = proto_dist_torch.detach().cpu().numpy()  # shape (B, P)
        occurrence_map = occurrence_map_torch.detach().cpu().numpy()  # shape (B, P, 1, (T), H, W)
        # pred = pred_torch.detach().cpu().numpy() # shape (B, num_classes)
        pred = logits.detach().cpu().numpy()  # shape (B, num_classes)
        gt = data_sample["target_AS"].detach().cpu().numpy()  # shape (B)
        image = x.detach().cpu().numpy()  # shape (B, 3, (To), Ho, Wo)
        filename = data_sample["filename"]  # shape (B)

        # for each prototype, find the minimum distance and their indices
        for j in range(P):
            proto_dist_j = proto_dist[:, j]  # (B)
            if proto_class_specific[j]:
                # compare with only the images of the prototype's class
                proto_dist_j = np.ma.masked_array(proto_dist_j, gt != proto_class_identity[j])
                if proto_dist_j.mask.all():
                    # if none of the classes this batch are the class of interest, move on
                    continue
            proto_dist_j_min = np.amin(proto_dist_j)  # scalar

            # if the distance this batch is smaller than prev.best, save it
            if proto_dist_j_min <= proto_dist_[j]:
                a = np.argmin(proto_dist_j)
                proto_dist_[j] = proto_dist_j_min
                protoL_input_[j] = protoL_input[a, j]
                occurrence_map_[j] = occurrence_map[a, j]
                pred_[j] = pred[a]
                image_[j] = image[a]
                gt_[j] = gt[a]
                filename_[j] = filename[a]

    prototypes_similarity_to_src_ROIs = 1 - np.array(proto_dist_)  # invert distance to similarity  shape (P)
    prototypes_occurrence_maps = np.array(occurrence_map_)  # shape (P, 1, (T), H, W)
    prototypes_src_imgs = np.array(image_)  # shape (P, 3, (To), Ho, Wo)
    prototypes_gts = np.array(gt_)  # shape (P)
    prototypes_preds = np.array(pred_)  # shape (P, K)
    prototypes_filenames = np.array(filename_)  # shape (P)

    # save the prototype information in a pickle file
    prototype_data_dict = {
        "prototypes_filenames": prototypes_filenames,
        "prototypes_src_imgs": prototypes_src_imgs,
        "prototypes_gts": prototypes_gts,
        "prototypes_preds": prototypes_preds,
        "prototypes_occurrence_maps": prototypes_occurrence_maps,
        "prototypes_similarity_to_src_ROIs": prototypes_similarity_to_src_ROIs,
    }
    save_pickle(prototype_data_dict, f"{proto_epoch_dir}/prototypes_info.pickle")

    # perform visualization for each prototype
    log("\tVisualizing prototypes ...")
    for j in range(P):
        if image_[j] is not None:
            prototype_plot(
                image_[j],
                occurrence_map_[j],
                j,
                filename_[j],
                pred_[j],
                gt_[j],
                proto_epoch_dir,
                interp="bilinear",
            )

    if replace_prototypes:
        protoL_input_ = np.array(protoL_input_)
        log("\tExecuting push ...")
        prototype_update = np.reshape(protoL_input_, tuple(prototype_shape))
        model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))
