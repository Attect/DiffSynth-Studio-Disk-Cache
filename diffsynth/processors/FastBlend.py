import os

import cupy as cp
import numpy
import numpy as np
from PIL import Image
from tqdm import tqdm

from .base import VideoProcessor
from ..extensions.FastBlend.patch_match import PyramidPatchMatcher
from ..extensions.FastBlend.runners.fast import TableManager


class FastBlendSmoother(VideoProcessor):
    def __init__(
            self,
            output_folder="",
            inference_mode="fast", batch_size=8, window_size=60,
            minimum_patch_size=5, threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0, initialize="identity", tracking_window_size=0
    ):
        self.cache_folder = os.path.join(output_folder, "smoother")
        os.makedirs(self.cache_folder, exist_ok=True)
        self.cache_folder_left = os.path.join(self.cache_folder, "left")
        os.makedirs(self.cache_folder_left, exist_ok=True)
        self.cache_folder_right = os.path.join(self.cache_folder, "right")
        os.makedirs(self.cache_folder_right, exist_ok=True)
        self.inference_mode = inference_mode
        self.batch_size = batch_size
        self.window_size = window_size
        self.ebsynth_config = {
            "minimum_patch_size": minimum_patch_size,
            "threads_per_block": threads_per_block,
            "num_iter": num_iter,
            "gpu_id": gpu_id,
            "guide_weight": guide_weight,
            "initialize": initialize,
            "tracking_window_size": tracking_window_size
        }

    @staticmethod
    def from_model_manager(model_manager, output_folder, **kwargs):
        # TODO: fetch GPU ID from model_manager
        return FastBlendSmoother(output_folder=output_folder, **kwargs)

    def inference_fast(self, frames_guide, frames_style):
        table_manager = TableManager()

        first_image = Image.open(frames_guide[0].tolist())
        patch_match_engine = PyramidPatchMatcher(
            image_height=first_image.height,
            image_width=first_image.width,
            channel=3,
            **self.ebsynth_config
        )
        # left part
        table_l = table_manager.build_remapping_table(self.cache_folder_left, frames_guide, frames_style, patch_match_engine, self.batch_size, desc="Fast Mode Step 1/4")
        table_l = table_manager.remapping_table_to_blending_table(self.cache_folder_left, table_l)
        table_l = table_manager.process_window_sum(self.cache_folder_left, frames_guide, table_l, patch_match_engine, self.window_size, self.batch_size, desc="Fast Mode Step 2/4")
        # right part
        table_r = table_manager.build_remapping_table(self.cache_folder_right, frames_guide[::-1], frames_style[::-1], patch_match_engine, self.batch_size, desc="Fast Mode Step 3/4")
        table_r = table_manager.remapping_table_to_blending_table(self.cache_folder_right, table_r)
        table_r = table_manager.process_window_sum(self.cache_folder_right, frames_guide[::-1], table_r, patch_match_engine, self.window_size, self.batch_size, desc="Fast Mode Step 4/4")[::-1]
        # merge
        frames = []
        index = 0
        for (frame_l, weight_l), frame_m, (frame_r, weight_r) in zip(table_l, frames_style, table_r):
            if isinstance(frame_l, str):
                frame_l = numpy.load(frame_l)
            if isinstance(frame_r, str):
                frame_r = numpy.load(frame_r)
            weight_m = -1
            weight = weight_l + weight_m + weight_r
            frame = frame_l * (weight_l / weight) + np.array(Image.open(frame_m.tolist())) * (weight_m / weight) + frame_r * (weight_r / weight)
            path = os.path.join(self.cache_folder, f"{index}.png")
            Image.fromarray(frame.clip(0, 255).astype("uint8")).save(path)
            frames.append(path)
            index += 1
        # frames = [frame.clip(0, 255).astype("uint8") for frame in frames]
        # frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def inference_balanced(self, frames_guide, frames_style):
        first_image = Image.open(frames_style[0].tolist())
        patch_match_engine = PyramidPatchMatcher(
            image_height=first_image.height,
            image_width=first_image.width,
            channel=3,
            **self.ebsynth_config
        )
        output_frames = []
        # tasks
        n = len(frames_style)
        tasks = []
        for target in range(n):
            for source in range(target - self.window_size, target + self.window_size + 1):
                if source >= 0 and source < n and source != target:
                    tasks.append((source, target))
        # run
        index = 0
        frames = [(None, 1) for i in range(n)]
        for batch_id in tqdm(range(0, len(tasks), self.batch_size), desc="Balanced Mode"):
            tasks_batch = tasks[batch_id: min(batch_id + self.batch_size, len(tasks))]
            source_guide = np.stack([np.array(Image.open(frames_guide[source].tolist())) for source, target in tasks_batch])
            target_guide = np.stack([np.array(Image.open(frames_guide[target].tolist())) for source, target in tasks_batch])
            source_style = np.stack([np.array(Image.open(frames_style[source].tolist())) for source, target in tasks_batch])
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for (source, target), result in zip(tasks_batch, target_style):
                frame, weight = frames[target]
                if frame is None:
                    frame = frames_style[target]
                elif isinstance(frame, str):
                    frame = numpy.load(frame)
                target_frame = frame * (weight / (weight + 1)) + result / (weight + 1)
                nyp_path = os.path.join(self.cache_folder, f"balance_{target}.nyp")
                numpy.save(nyp_path, target_frame)
                frames[target] = (
                    nyp_path,
                    weight + 1
                )
                if weight + 1 == min(n, target + self.window_size + 1) - max(0, target - self.window_size):
                    frame = frame.clip(0, 255).astype("uint8")
                    path = os.path.join(self.cache_folder, f"{index}.png")
                    Image.fromarray(frame).save(path)
                    output_frames.append(path)
                    frames[target] = (None, 1)
                    index += 1
        return output_frames

    def inference_accurate(self, frames_guide, frames_style):
        global patch_match_engine
        first_image = Image.open(frames_style[0].tolist())
        patch_match_engine = PyramidPatchMatcher(
            image_height=first_image.height,
            image_width=first_image.width,
            channel=3,
            use_mean_target_style=True,
            **self.ebsynth_config
        )
        output_frames = []
        index = 0
        # run
        n = len(frames_style)
        for target in tqdm(range(n), desc="Accurate Mode"):
            l, r = max(target - self.window_size, 0), min(target + self.window_size + 1, n)
            remapped_frames = []
            for i in range(l, r, self.batch_size):
                j = min(i + self.batch_size, r)
                source_guide = np.stack([np.array(Image.open(frames_guide[source].tolist())) for source in range(i, j)])
                target_guide = np.stack([np.array(Image.open(frames_guide[target].tolist()))] * (j - i))
                source_style = np.stack([np.array(Image.open(frames_style[source].tolist())) for source in range(i, j)])
                _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
                remapped_frames.append(target_style)
            frame = np.concatenate(remapped_frames, axis=0).mean(axis=0)
            path = os.path.join(self.cache_folder, f"{index}.png")
            Image.fromarray(frame.clip(0, 255).astype("uint8")).save(path)
            output_frames.append(path)
            index += 1
        return output_frames

    def release_vram(self):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    def __call__(self, rendered_frames, original_frames=None, **kwargs):
        rendered_frames = [np.array(frame) for frame in rendered_frames]
        original_frames = [np.array(frame) for frame in original_frames]
        if self.inference_mode == "fast":
            output_frames = self.inference_fast(original_frames, rendered_frames)
        elif self.inference_mode == "balanced":
            output_frames = self.inference_balanced(original_frames, rendered_frames)
        elif self.inference_mode == "accurate":
            output_frames = self.inference_accurate(original_frames, rendered_frames)
        else:
            raise ValueError("inference_mode must be fast, balanced or accurate")
        self.release_vram()
        return output_frames
