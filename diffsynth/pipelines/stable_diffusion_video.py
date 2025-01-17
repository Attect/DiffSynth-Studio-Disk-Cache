import json
import os
import shutil
import time
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .dancer import lets_dance
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..data import VideoData, save_frames, save_video
from ..models import ModelManager, SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder, SDMotionModel
from ..processors.sequencial_processor import SequencialProcessor
from ..prompts import SDPrompter
from ..schedulers import EnhancedDDIMScheduler


def lets_dance_with_long_video(
        unet: SDUNet,
        motion_modules: SDMotionModel = None,
        controlnet: MultiControlNetManager = None,
        sample=None,
        timestep=None,
        encoder_hidden_states=None,
        controlnet_processor_count=2,
        animatediff_batch_size=16,
        animatediff_stride=8,
        unet_batch_size=1,
        controlnet_batch_size=1,
        controlnet_cache_dir="controlnet_cache",
        cross_frame_attention=False,
        device="cuda",
        vram_limit_level=0,
):
    num_frames = sample.shape[0]
    hidden_states_output = [(torch.zeros(sample[0].shape, dtype=sample[0].dtype), 0) for i in range(num_frames)]

    controlnet_hold_cache = {}

    for batch_id in tqdm(range(0, num_frames, animatediff_stride), leave=False, desc=f'dance_batch'):
        batch_id_ = min(batch_id + animatediff_batch_size, num_frames)

        stack_controlnet_file_contents = []
        process_caches = []
        controlnet_cache_frames = None

        for processor_id in range(controlnet_processor_count):
            controlnet_file_contents = []
            for i in range(batch_id, batch_id_):
                cache_path = controlnet_cache_dir + f'/cache_p{processor_id}_{i}.pt'
                if cache_path in controlnet_hold_cache:
                    load_data = controlnet_hold_cache[cache_path]
                    # print(f'命中缓存{cache_path}')
                else:
                    load_data = torch.load(cache_path)[0]
                    controlnet_hold_cache[cache_path] = load_data
                    # print(f'加载缓存{cache_path}')
                    if len(controlnet_hold_cache) > animatediff_batch_size * 2:
                        cache_key = next(iter(controlnet_hold_cache))
                        if cache_key is not cache_path:
                            controlnet_hold_cache.pop(cache_key)
                            # print(f'释放缓存{cache_key}')
                controlnet_file_contents.append(load_data)
            process_caches.append(torch.stack(controlnet_file_contents, dim=0))

        if controlnet_processor_count > 0:
            stack_controlnet_file_contents.append(torch.stack(process_caches, dim=0))
            controlnet_cache_frames = torch.cat(stack_controlnet_file_contents, dim=0)

        # process this batch
        hidden_states_batch = lets_dance(
            unet, motion_modules, controlnet,
            sample[batch_id: batch_id_].to(device),
            timestep,
            encoder_hidden_states[batch_id: batch_id_].to(device),
            controlnet_cache_frames.to(device) if controlnet_processor_count > 0 else None,
            unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
            cross_frame_attention=cross_frame_attention,
            device=device, vram_limit_level=vram_limit_level
        ).cpu()

        # update hidden_states
        for i, hidden_states_updated in zip(range(batch_id, batch_id_), hidden_states_batch):
            bias = max(1 - abs(i - (batch_id + batch_id_ - 1) / 2) / ((batch_id_ - batch_id - 1 + 1e-2) / 2), 1e-2)
            hidden_states, num = hidden_states_output[i]
            hidden_states = hidden_states * (num / (num + bias)) + hidden_states_updated * (bias / (num + bias))
            hidden_states_output[i] = (hidden_states, num + bias)

        if batch_id_ == num_frames:
            break

    # output
    hidden_states = torch.stack([h for h, _ in hidden_states_output])
    return hidden_states


class SDVideoPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16, use_animatediff=True):
        super().__init__()
        self.scheduler = EnhancedDDIMScheduler(beta_schedule="linear" if use_animatediff else "scaled_linear")
        self.prompter = SDPrompter()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.text_encoder: SDTextEncoder = None
        self.unet: SDUNet = None
        self.vae_decoder: SDVAEDecoder = None
        self.vae_encoder: SDVAEEncoder = None
        self.controlnet: MultiControlNetManager = None
        self.motion_modules: SDMotionModel = None

    def fetch_main_models(self, model_manager: ModelManager):
        self.text_encoder = model_manager.text_encoder
        self.unet = model_manager.unet
        self.vae_decoder = model_manager.vae_decoder
        self.vae_encoder = model_manager.vae_encoder

    def fetch_controlnet_models(self, model_manager: ModelManager,
                                controlnet_config_units: List[ControlNetConfigUnit] = []):
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id),
                model_manager.get_model_with_model_path(config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = MultiControlNetManager(controlnet_units)

    def fetch_motion_modules(self, model_manager: ModelManager):
        if "motion_modules" in model_manager.model:
            self.motion_modules = model_manager.motion_modules

    def fetch_prompter(self, model_manager: ModelManager):
        self.prompter.load_from_model_manager(model_manager)

    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit] = []):
        pipe = SDVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
            use_animatediff="motion_modules" in model_manager.model
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_motion_modules(model_manager)
        pipe.fetch_prompter(model_manager)
        pipe.fetch_controlnet_models(model_manager, controlnet_config_units)
        return pipe

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(torch.float32).to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        # if not np.isfinite(image).all():
        #     return None
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image

    def decode_images(self, latents, output_folder, tiled=False, tile_size=64, tile_stride=32):
        cache_dir = os.path.join(output_folder, "latents")
        os.makedirs(cache_dir, exist_ok=True)

        result = []
        for frame_id in tqdm(range(latents.shape[0]), desc="VAE Decode"):
            save_path = os.path.join(cache_dir, f'image_{frame_id}.png')
            image = self.decode_image(latents[frame_id: frame_id + 1], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if image is None:
                print(f"latent failed at {frame_id} , try smaller tile_size")
                tile_size = 32
                image = self.decode_image(latents[frame_id: frame_id + 1], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if image is not None:
                image.save(save_path)
                result.append(save_path)
            else:
                print(f"latent failed at {frame_id} , all try failed, saved latents.pt")
                latent_path = cache_dir + "/latents.pt"
                torch.save(latents, latent_path)
                break
        return result
        # images = [
        #     self.decode_image(latents[frame_id: frame_id + 1], tiled=tiled, tile_size=tile_size,
        #                       tile_stride=tile_stride)
        #     for frame_id in range(latents.shape[0])
        # ]
        # return images

    def encode_images(self, processed_images, tiled=False, tile_size=64, tile_stride=32):
        latents = []
        for image in processed_images:
            if isinstance(image, str):
                image = Image.open(image)
            image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
            latent = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).cpu()
            latents.append(latent)

        latents = torch.concat(latents, dim=0)
        return latents

    @torch.no_grad()
    def __call__(
            self,
            prompt,
            negative_prompt="",
            cfg_scale=7.5,
            clip_skip=1,
            num_frames=None,
            input_frames=None,
            controlnet_frames=None,
            denoising_strength=1.0,
            height=512,
            width=512,
            num_inference_steps=20,
            animatediff_batch_size=16,
            animatediff_stride=8,
            unet_batch_size=1,
            controlnet_batch_size=1,
            cross_frame_attention=False,
            smoother=None,
            smoother_progress_ids=[],
            vram_limit_level=0,
            progress_bar_cmd=tqdm,
            progress_bar_st=None,
            clear_output_folder=False,
            output_folder="output",
    ):
        # Prepare controlnet cacheDir

        controlnet_cache_dir = os.path.join(output_folder, "controlnet_caches")
        os.makedirs(controlnet_cache_dir, exist_ok=True)

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if self.motion_modules is None:
            noise = torch.randn((1, 4, height // 8, width // 8), device="cpu", dtype=self.torch_dtype).repeat(
                num_frames, 1, 1, 1)
        else:
            noise = torch.randn((num_frames, 4, height // 8, width // 8), device="cpu", dtype=self.torch_dtype)
        if input_frames is None or denoising_strength == 1.0:
            latents = noise
        else:
            latents = self.encode_images(input_frames)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])

        # Encode prompts
        prompt_emb_posi = self.prompter.encode_prompt(self.text_encoder, prompt, clip_skip=clip_skip,
                                                      device=self.device, positive=True).cpu()
        prompt_emb_nega = self.prompter.encode_prompt(self.text_encoder, negative_prompt, clip_skip=clip_skip,
                                                      device=self.device, positive=False).cpu()
        prompt_emb_posi = prompt_emb_posi.repeat(num_frames, 1, 1)
        prompt_emb_nega = prompt_emb_nega.repeat(num_frames, 1, 1)

        controlnet_processor_count = self.controlnet.unit_count() if controlnet_frames is not None else 0

        # Prepare ControlNets
        if controlnet_frames is not None:
            if isinstance(controlnet_frames[0], list):
                for processor_id in range(len(controlnet_frames)):
                    index = 0
                    for controlnet_frame in progress_bar_cmd(controlnet_frames[processor_id], desc=f'make_controlnet_processor_{processor_id}_cache'):
                        cache_full_path = controlnet_cache_dir + f'/cache_p{processor_id}_{index}.pt'
                        if not os.path.exists(cache_full_path):
                            tensor_data = self.controlnet.process_image(controlnet_frame, processor_id=processor_id).to(self.torch_dtype)
                            torch.save(tensor_data, cache_full_path)
                        index += 1
            else:
                controlnet_processor_count = 1
                index = 0
                for controlnet_frame in progress_bar_cmd(controlnet_frames):
                    cache_full_path = controlnet_cache_dir + f'/cache_p0_{index}.pt'
                    if not os.path.exists(cache_full_path):
                        tensor_data = self.controlnet.process_image(controlnet_frame).to(self.torch_dtype)
                        torch.save(tensor_data, cache_full_path)
                    index += 1
                # controlnet_frames = torch.stack([
                #     self.controlnet.process_image(controlnet_frame).to(self.torch_dtype)
                #     for controlnet_frame in progress_bar_cmd(controlnet_frames)
                # ], dim=1)

        # Denoise
        save_process_id_path = output_folder + "/last_process_id.txt"
        saved_process_id = -1
        if os.path.exists(save_process_id_path):
            with open(save_process_id_path) as f:
                saved_process_id = int(f.read())

        cache_latents_path = output_folder + '/latents.py'
        if saved_process_id > -1 and os.path.exists(cache_latents_path):
            latents = torch.load(cache_latents_path)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(self.device)
            if saved_process_id > -1 and progress_id <= saved_process_id:
                print(f'\n根据保存进度{saved_process_id}，跳过{progress_id}')
                time.sleep(1)
                continue

            # Classifier-free guidance
            noise_pred_posi = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_posi,
                controlnet_processor_count=controlnet_processor_count,
                animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
                unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
                cross_frame_attention=cross_frame_attention,
                controlnet_cache_dir=controlnet_cache_dir,
                device=self.device, vram_limit_level=vram_limit_level
            )
            noise_pred_nega = lets_dance_with_long_video(
                self.unet, motion_modules=self.motion_modules, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_nega,
                controlnet_processor_count=controlnet_processor_count,
                animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
                unet_batch_size=unet_batch_size, controlnet_batch_size=controlnet_batch_size,
                cross_frame_attention=cross_frame_attention,
                controlnet_cache_dir=controlnet_cache_dir,
                device=self.device, vram_limit_level=vram_limit_level
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM and smoother
            if smoother is not None and progress_id in smoother_progress_ids:
                rendered_frames = self.scheduler.step(noise_pred, timestep, latents, to_final=True)
                rendered_frames = self.decode_images(rendered_frames, output_folder)
                rendered_frames = smoother(rendered_frames, original_frames=input_frames)
                target_latents = self.encode_images(rendered_frames)
                noise_pred = self.scheduler.return_to_timestep(timestep, latents, target_latents)
            latents = self.scheduler.step(noise_pred, timestep, latents)

            with open(save_process_id_path, 'w') as f:
                f.write(str(progress_id))
            torch.save(latents, cache_latents_path)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        output_frames = self.decode_images(latents, output_folder)

        # Post-process
        if smoother is not None and (num_inference_steps in smoother_progress_ids or -1 in smoother_progress_ids):
            output_frames = smoother(output_frames, original_frames=input_frames)

        return output_frames


class SDVideoPipelineRunner:
    def __init__(self, in_streamlit=False):
        self.in_streamlit = in_streamlit

    def load_pipeline(self, model_list, textual_inversion_folder, device, lora_alphas, controlnet_units):
        # Load models
        model_manager = ModelManager(torch_dtype=torch.float16, device=device)
        model_manager.load_textual_inversions(textual_inversion_folder)
        model_manager.load_models(model_list, lora_alphas=lora_alphas)
        pipe = SDVideoPipeline.from_model_manager(
            model_manager,
            [
                ControlNetConfigUnit(
                    processor_id=unit["processor_id"],
                    model_path=unit["model_path"],
                    scale=unit["scale"]
                ) for unit in controlnet_units
            ]
        )
        return model_manager, pipe

    def load_smoother(self, model_manager, output_folder, smoother_configs):
        smoother = SequencialProcessor.from_model_manager(model_manager, output_folder, smoother_configs)
        return smoother

    def synthesize_video(self, model_manager, pipe, seed, smoother, **pipeline_inputs):
        torch.manual_seed(seed)
        if self.in_streamlit:
            import streamlit as st
            progress_bar_st = st.progress(0.0)
            output_video = pipe(**pipeline_inputs, smoother=smoother, progress_bar_st=progress_bar_st)
            progress_bar_st.progress(1.0)
        else:
            output_video = pipe(**pipeline_inputs, smoother=smoother)
        model_manager.to("cpu")
        return output_video

    def load_video(self, video_file, image_folder, output_folder, height, width, start_frame_id, end_frame_id):
        image_cache_folder = os.path.join(output_folder, "source_images")
        os.makedirs(image_cache_folder, exist_ok=True)
        video = VideoData(video_file=video_file, image_folder=image_folder, image_cache_folder=image_cache_folder, height=height, width=width)
        if start_frame_id is None:
            start_frame_id = 0
        if end_frame_id is None:
            end_frame_id = len(video)
        frames = [video[i] for i in tqdm(range(start_frame_id, end_frame_id), desc="Decode Images")]
        return frames

    def add_data_to_pipeline_inputs(self, data, pipeline_inputs):
        pipeline_inputs["input_frames"] = self.load_video(**data["input_frames"], output_folder=data["output_folder"])
        pipeline_inputs["num_frames"] = len(pipeline_inputs["input_frames"])
        pipeline_inputs["width"], pipeline_inputs["height"] = Image.open(pipeline_inputs["input_frames"][0]).size
        pipeline_inputs["clear_output_folder"] = data["clear_output_folder"]
        pipeline_inputs["output_folder"] = data["output_folder"]
        if len(data["controlnet_frames"]) > 0:
            pipeline_inputs["controlnet_frames"] = [self.load_video(**unit, output_folder=data["output_folder"]) for unit in data["controlnet_frames"]]
        return pipeline_inputs

    def save_output(self, video, output_folder, fps, config):
        os.makedirs(output_folder, exist_ok=True)
        save_frames(video, os.path.join(output_folder, "frames"))
        save_video(video, os.path.join(output_folder, "video.mp4"), fps=fps)
        config["pipeline"]["pipeline_inputs"]["input_frames"] = []
        config["pipeline"]["pipeline_inputs"]["controlnet_frames"] = []
        with open(os.path.join(output_folder, "config.json"), 'w') as file:
            json.dump(config, file, indent=4)

    def run(self, config):
        output_folder = config["data"]["output_folder"]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        elif config["data"]["clear_output_folder"]:
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        if self.in_streamlit:
            import streamlit as st
        if self.in_streamlit: st.markdown("Loading videos ...")
        config["pipeline"]["pipeline_inputs"] = self.add_data_to_pipeline_inputs(config["data"],
                                                                                 config["pipeline"]["pipeline_inputs"])
        if self.in_streamlit: st.markdown("Loading videos ... done!")
        if self.in_streamlit: st.markdown("Loading models ...")
        model_manager, pipe = self.load_pipeline(**config["models"])
        if self.in_streamlit: st.markdown("Loading models ... done!")
        if "smoother_configs" in config:
            if self.in_streamlit: st.markdown("Loading smoother ...")
            smoother = self.load_smoother(model_manager, output_folder=output_folder, smoother_configs=config["smoother_configs"])
            if self.in_streamlit: st.markdown("Loading smoother ... done!")
        else:
            smoother = None
        if self.in_streamlit: st.markdown("Synthesizing videos ...")
        output_video = self.synthesize_video(model_manager, pipe, config["pipeline"]["seed"], smoother,
                                             **config["pipeline"]["pipeline_inputs"])
        if self.in_streamlit: st.markdown("Synthesizing videos ... done!")
        if self.in_streamlit: st.markdown("Saving videos ...")
        self.save_output(output_video, config["data"]["output_folder"], config["data"]["fps"], config)
        if self.in_streamlit: st.markdown("Saving videos ... done!")
        if self.in_streamlit: st.markdown("Finished!")
        # video_file = open(os.path.join(os.path.join(config["data"]["output_folder"], "video.mp4")), 'rb')
        # if self.in_streamlit: st.video(video_file.read())
