import numpy as np
import matplotlib.pyplot as plt
import os

from src.utils.video_utils import saveVideo
from src.utils.explainability_utils import (
    load_data_and_model_products,
    get_src,
    get_normalized_upsample_occurence_maps,
    get_heatmap,
)
from src.utils.utils import load_pickle, save_pickle
from tqdm import tqdm
from moviepy.video.io.bindings import mplfig_to_npimage


def explain_local(
    mode,  # val or test
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    model,  # pytorch network with prototype_vectors
    data_config,
    abstain_class=True,  # indicates K+1-th class is of the "abstain" type
    model_directory=None,  # if not None, explainability results will be saved here
    epoch_number=0,
    log=print,
):
    if model_directory is not None:
        root_dir_for_saving = os.path.join(model_directory, f"epoch_{epoch_number}")
    else:
        root_dir_for_saving = f"local_explain_model_epoch_{epoch_number}"

    model.eval()
    log(f"\t local explanation of model in {root_dir_for_saving}")

    ##### Loading the prototype information (src img, heatmap, etc)
    prots_info_path = os.path.join(model_directory, f"img/epoch-{epoch_number}_pushed/prototypes_info.pickle")
    if os.path.exists(prots_info_path):
        prots_data_dict = load_pickle(prots_info_path, log)
    else:
        raise f"path {prots_info_path} does not exist. Project the prototypes with Push function first"

    ##### Loading/creating the validation/training data and model products
    data_dict, model_products_dict = load_data_and_model_products(
        model,
        dataloader,
        mode,
        data_config,
        abstain_class,
        root_dir_for_saving,
        log=log,
    )

    n_prototypes = model.num_prototypes
    num_classes = model.num_classes
    fc_layer_weights = model_products_dict["fc_layer_weights"]
    similarities_ = 1 - model_products_dict["proto_dist_"]
    n_prots_per_class = n_prototypes // num_classes

    #########################################
    ##### Process the prototype info ########
    # get the source image/video
    prots_src_imgs, upsampler = get_src(prots_data_dict["prototypes_src_imgs"])  # shape (P, (To), Ho, Wo, 3)

    # resize, upsample, and normalize the occurrence map.  shape (P, (To), Ho, Wo)
    prots_occurrence_maps = prots_data_dict["prototypes_occurrence_maps"]  # shape (P, 1, (T), H, W)
    prots_rescaled_occurrence_maps = get_normalized_upsample_occurence_maps(
        prots_occurrence_maps, upsampler
    )  # shape (P, (T), H, W)

    # # prototype src image masked with normalized occurrence map
    # mask = np.expand_dims(prots_rescaled_occurrence_maps, axis=-1)
    # prots_src_imgs_masked = prots_src_imgs * mask

    # prototype src image with normalized occurrence map overlay
    prots_heatmaps = get_heatmap(prots_rescaled_occurrence_maps)  # shape (P, (To), Ho, Wo, 3)
    prots_overlayed_imgs = prots_src_imgs + 0.3 * prots_heatmaps  # shape (P, (To), Ho, Wo, 3)

    ######################################################################
    ##### loop over the dataset to locally explain each datapoint ########
    iterator = tqdm(range(len(data_dict["filenames"])), dynamic_ncols=True)
    for i in iterator:
        #########################################
        ##### get the img's information
        test_filename = data_dict["filenames"][i]
        src_img = data_dict["inputs"][i : i + 1]  # shape = (N=1, 3, To, Ho, Wo)
        gt = data_dict["ys_gt"][i]
        ##### get the model products for the img
        occurrence_maps = model_products_dict["occurrence_map_"][i]  # shape (P, 1, (T), H, W)
        protoL_input = model_products_dict["protoL_input_"][i]  # shape = (n_prototypes, Channel)
        similarities = similarities_[i]  # shape = (n_prototypes)
        contributions = np.matmul(np.expand_dims(similarities, axis=0), fc_layer_weights.T)[0]
        pred = model_products_dict["ys_pred"][i]  # shape = (num_classes)
        ##################################################################
        ##### Process the info to get src images/videos and overlays #####
        # get the source image
        test_image_raw, upsampler = get_src(src_img)  # shape test_image_raw (1, (To), Ho, Wo, 3)
        test_image_raw = test_image_raw[0]  # to remove the batch dimension shape ((To), Ho, Wo, 3)

        # resize, upsample, and normalize the occurrence map.  shape (P, (To), Ho, Wo)
        rescaled_occurrence_maps = get_normalized_upsample_occurence_maps(occurrence_maps, upsampler)

        ##### image with normalized occurrence map overlay
        heatmaps = get_heatmap(rescaled_occurrence_maps)
        test_image_overlays = test_image_raw + 0.3 * heatmaps  # shape (P, (To), Ho, Wo, 3)
        ##### image masked with normalized occurrence map
        # masks = np.expand_dims(rescaled_occurrence_maps, axis=-1)
        # test_image_masks = test_image_raw * masks

        ########################################################################################
        ##### Plot and save the test image/video and its similarity to each prototype separately #####
        # plot all sorted based on their similarity score
        # sort the prototypes for each class
        sorted_similarities_indices = []
        sorted_similarities = []
        for class_indx in range(num_classes):
            sorted_indices = np.argsort(
                similarities[class_indx * n_prots_per_class : (1 + class_indx) * n_prots_per_class]
            )
            sorted_indices = sorted_indices[..., ::-1] + n_prots_per_class * class_indx
            sorted_similarities_indices.extend(sorted_indices)

            sorted_sims = np.sort(similarities[class_indx * n_prots_per_class : (1 + class_indx) * n_prots_per_class])[
                ..., ::-1
            ]
            sorted_similarities.extend(sorted_sims)

        format = "gif"  # gif or mp4
        save_path = os.path.join(root_dir_for_saving, mode, "local", test_filename)
        saveVideo(test_image_raw * 255, save_path, f"test_clip_AS-{gt}", format=format, fps=10)
        os.makedirs(save_path, exist_ok=True)
        # # Plot each prototype separately
        # for p in range(n_prototypes):
        #     saveVideo(test_image_overlays[p]*255, save_path, f'AS-{p//n_prots_per_class}_{similarities[p]:.2f}_{p:02d}',
        #               gif=True, avi=False, fps=10)

        #### PLOTTING
        # Plot each prototype separately
        prot_iterator = tqdm(range(n_prototypes), dynamic_ncols=True)
        for p in prot_iterator:
            ####################### original echo and prototype together #################################
            save_filename = f"AS-{p//n_prots_per_class}_{similarities[p]:.2f}_{p:02d}"
            file_path = f"{save_path}/{save_filename}.{format}"
            if not os.path.exists(file_path):
                frames = []
                frame_iterator = tqdm(range(test_image_raw.shape[0]), dynamic_ncols=True)
                for t in frame_iterator:
                    fig, axs = plt.subplots(1, 4, figsize=(12, 5))
                    [axi.set_axis_off() for axi in axs.ravel()]

                    # 1. Test Case Raw Images with Label and Predictions of the model
                    axs[0].imshow(test_image_raw[t])
                    axs[0].title.set_text(f"Test Case")

                    # 2. Test Image Overlay of ROI Heatmap for Prototype "p", Heatmap Normalized Across the Video/Image
                    axs[1].imshow(test_image_overlays[p, t])
                    axs[1].title.set_text(f"Similarity to p_{p:d}={similarities[p]:.2f}")

                    # 3. Prototype with Its Heatmap Overlay, Heatmap Normalized Across the Video/Image
                    axs[2].imshow(prots_overlayed_imgs[p, t])
                    importance = fc_layer_weights[:, p]
                    contribution = importance * similarities[p]
                    axs[2].title.set_text(
                        # f"Importance = {[f'{importance[c]:.2f}' for c in range(gt.shape[0])]}\n"
                        f"        Contibution       \n"
                        f"{[f'{int(contribution[c]*100)}/{int(contributions[c]*100)}' for c in range(pred.shape[0])]}"
                        # f"{[f'{contribution[c]:.2f}/{contributions[c]:.2f}' for c in range(pred.shape[0])]}"
                    )

                    # 4. Prototype Raw Images
                    axs[3].imshow(prots_src_imgs[p, t])
                    axs[3].title.set_text(f"Rank-{list(sorted_similarities_indices).index(p) % 10} | p_{p:d}")

                    # add super title for the figure
                    fig.suptitle(
                        f"p_{p:02d}  | img_pred = {[f'{pred[c]:.2f}' for c in range(pred.shape[0])]} | gt = {gt}",
                        fontsize=15,
                    )
                    # some visual configs for the figure
                    fig.tight_layout()

                    frame = mplfig_to_npimage(fig)  # shape (H, W, 3)
                    frames.append(frame)

                    plt.close(fig)

                frames = np.asarray(frames)  # shape ((To), H, W, 3)
                # saveVideo(frames, save_path, save_filename, gif=True, avi=True, fps=10)
                saveVideo(frames, save_path, save_filename, format=format, fps=10)

            ####################### prototype with its heatmap overlaid #################################
            os.makedirs(f"{save_path}/prototype_overlaid", exist_ok=True)
            file_path = f"{save_path}/prototype_overlaid/{save_filename}.{format}"
            if not os.path.exists(file_path):
                frames = []
                frame_iterator = tqdm(range(test_image_raw.shape[0]), dynamic_ncols=True)
                for t in frame_iterator:
                    fig = plt.figure(figsize=(6, 6))

                    # Prototype with Its Heatmap Overlay, Heatmap Normalized Across the Video/Image
                    plt.imshow(prots_overlayed_imgs[p, t])

                    # some visual configs for the figure
                    plt.axis("off")
                    fig.tight_layout()

                    frame = mplfig_to_npimage(fig)  # shape (H, W, 3)
                    frames.append(frame)

                    plt.close(fig)

                frames = np.asarray(frames)  # shape ((To), H, W, 3)
                # saveVideo(frames, save_path, save_filename, gif=True, avi=True, fps=10)
                saveVideo(
                    frames,
                    f"{save_path}/prototype_overlaid",
                    save_filename,
                    format=format,
                    fps=10,
                )

            ####################### Input echo with its heatmap overlaid #################################
            os.makedirs(f"{save_path}/input_overlaid", exist_ok=True)
            file_path = f"{save_path}/input_overlaid/{save_filename}.{format}"
            if not os.path.exists(file_path):
                frames = []
                frame_iterator = tqdm(range(test_image_raw.shape[0]), dynamic_ncols=True)
                for t in frame_iterator:
                    fig = plt.figure(figsize=(6, 6))

                    # 2. Test Image Overlay of ROI Heatmap for Prototype "p", Heatmap Normalized Across the Video/Image
                    plt.imshow(test_image_overlays[p, t])

                    # some visual configs for the figure
                    plt.axis("off")
                    fig.tight_layout()

                    frame = mplfig_to_npimage(fig)  # shape (H, W, 3)
                    frames.append(frame)

                    plt.close(fig)

                frames = np.asarray(frames)  # shape ((To), H, W, 3)
                # saveVideo(frames, save_path, save_filename, gif=True, avi=True, fps=10)
                saveVideo(
                    frames,
                    f"{save_path}/input_overlaid",
                    save_filename,
                    format=format,
                    fps=10,
                )
