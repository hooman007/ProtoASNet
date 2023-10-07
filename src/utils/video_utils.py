import moviepy.video.io.ImageSequenceClip
import imageio
import glob
import os
import os.path
from moviepy.editor import ImageSequenceClip


def write_video(img_list, output_path="my_video.mp4", fps=10):
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(img_list, fps=fps)
    clip.write_videofile(output_path)


def write_gif(img_list, output_path="my_gif.gif"):
    with imageio.get_writer(output_path, mode="I") as writer:
        for i in img_list:
            writer.append_data(imageio.imread(i))


def remove_images(img_list):
    for i in img_list:
        os.remove(i)


def saveVideo(sample, save_path, filename, format="mp4", fps=10):
    """

    :param sample: video with shape (T,H,W,3)
    :param save_path: directory to store the video
    :param filename: filename for the video saved
    :param type: format, either gif or mp4
    :return:
    """
    # ##### GIF ##########
    file_path = f"{save_path}/{filename}.{format}"
    if os.path.exists(file_path):
        return

    os.makedirs(save_path, exist_ok=True)
    clip = ImageSequenceClip(list(sample), fps=fps)  # input to clip=list of [H,W,3]
    if format == "gif":
        clip.write_gif(file_path, fps=fps)
    elif format == "mp4":
        clip.write_videofile(file_path, fps=fps, codec="mpeg4", verbose=False)
    else:
        raise f"The format {format} for saving video data is not supported!"
