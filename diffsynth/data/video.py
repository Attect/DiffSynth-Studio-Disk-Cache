import os
import shutil

import imageio
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)

    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()


def split_file_name(file_name):
    result = []
    number = -1
    for i in file_name:
        if ord(i) >= ord("0") and ord(i) <= ord("9"):
            if number == -1:
                number = 0
            number = number * 10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result = tuple(result)
    return result


def search_for_images(folder):
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


class LowMemoryImageFolder:
    def __init__(self, folder, file_list=None):
        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        return Image.open(self.file_list[item]).convert("RGB")

    def __del__(self):
        pass


def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left + croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left: left + croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image


def resize_and_fill(image, height, width):
    # 首先，保持比例缩放图像
    image.thumbnail((width, height), Image.BICUBIC)

    # 获取缩放后的图像尺寸
    image_width, image_height = image.size

    # 如果宽度小于目标宽度，则在左右边缘填充像素
    if image_width < width:
        left_pad = image.crop((0, 0, 1, image_height))
        right_pad = image.crop((image_width - 1, 0, image_width, image_height))
        left_padding = (width - image_width) // 2
        for i in range(left_padding):
            image = ImageOps.expand(image, (1, 0, 1, 0))
            image.paste(left_pad, (0, 0))
            image.paste(right_pad, (image_width - 1, 0))
            image_width = image_width + 2
        if image_width % 2 != 0:
            image = ImageOps.expand(image, (0, 0, 1, 0))
            image.paste(right_pad, (image_width, 0))

    # 如果高度小于目标高度，则在上下边缘填充像素
    if image_height < height:
        top_pad = image.crop((0, 0, image_width, 1))
        bottom_pad = image.crop((0, image_height - 1, image_width, image_height))
        top_padding = (height - image_height) // 2
        for i in range(top_padding):
            image = ImageOps.expand(image, (0, 1, 0, 1))
            image.paste(top_pad, (0, 0))
            image.paste(bottom_pad, (0, image_height - 1))
            image_height = image_height + 2
        if image_height % 2 != 0:
            image = ImageOps.expand(image, (0, 0, 0, 1))
            image.paste(bottom_pad, (0, image_height))
    return image

class VideoData:
    def __init__(self, video_file=None, image_folder=None, image_cache_folder="image_cache", height=None, width=None, **kwargs):
        self.cache_folder = image_cache_folder
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder, exist_ok=True)
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        path = f"{self.cache_folder}/{item}.png"
        if os.path.exists(path):
            return path
        frame = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = resize_and_fill(frame, self.height, self.width)
        frame.save(path)
        return path

    def __del__(self):
        pass

    def save_images(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            frame = self.__getitem__(i)
            frame.save(os.path.join(folder, f"{i}.png"))


def save_video(frames, save_path, fps, quality=9):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in tqdm(frames, desc="Saving video"):
        if isinstance(frame, str):
            frame = np.array(Image.open(frame))
        writer.append_data(np.array(frame))
    writer.close()


def save_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        shutil.copy(frame, os.path.join(save_path, f"{i}.png"))
        # frame.save(os.path.join(save_path, f"{i}.png"))
