# DiffSynth Studio 磁盘缓存版(Disk cache)

> 此仓库提供具有磁盘缓存效果的DiffySynth Studio
>
> 一些说明文档经过了AI翻译及人工校对

## DiffSynth Studio 磁盘缓存版特别说明

**此版本为个人所需[Issues 22](https://github.com/Artiprocher/DiffSynth-Studio/issues/22)**，根据原项目[DiffSynth-Studio](https://github.com/Artiprocher/DiffSynth-Studio)
，将运行过程中部分内存数据转移至磁盘，以节省内存需求（128GiB跑不动2分钟一镜到底的视频太离谱了）。

**此版本增加了自动继续上次进度的功能**，应对跑了几天却因为一些情况意外中断导致计算浪费的情况。

**此版本使用32位精度处理VAE解码**，以处理一些模型丢失部分帧的问题。

*注意：python菜鸟，修改此项目前不会python语言，粗暴糊了一坨磁盘缓存代码上去，且没有缓存管理和缓存加速机制（目前懒得设计，改现在的效果已经花了太多时间了），代码洁癖者慎看，并且完全不了解AI相关开发及运行原理，仅根据其它语言开发经验修改。已尽可能保证examples都可跑*
，有错可以提issuse。

### 磁盘缓存效果

使用`examples/diffutoon_toon_shading.py`修改参数，增加smoother，一次性处理3667帧，生成分辨率1280x768。
全流程内存使用：小于6GiB！！
磁盘储存使用：260GiB (可以理解为如果不增加这个磁盘缓存效果将需要260GiB的内存？或许没那么多但肯定也少不了)

### 自动继续上次进度

只能继承大进度，整体项目划分为10个步骤的精度。进度储存在输出目录的`latents.py`中，记录在输出目录的`last_process_id.txt`中，如果只想使用图像和controlnet的缓存重新跑，删除输出目录的这两个文件即可。

### 输出目录说明

> 输出目录增加了一些

增加了一个参数：`config.data.clear_output_folder`，若为True，则会清空输出目录，否则会利用上输出目录中（如果）存在的裁剪好的数据以及controlnet处理好的缓存以加速部分步骤（多次调整参数时可以提升效率）。

1. source_images：存放视频解码为图片流后，裁剪后的图片。
1. controlnet_caches：存放control_net的中间缓存数据。
1. latents：存放Stable Diffusion处理后的所有结果，png图片，可以拦截到smoother处理前的图像序列。
1. smoother：内部还有left及right，分别为FastBlend左右帧参考的处理结果，目录下的result开头的为左右及原帧参考结果。

### 速度

> 有时候，快还是慢不是问题，能不能才是关键，大硬盘和大内存，还是硬盘好搞。

测试机器：
CPU:AMD 7950x
GPU:Nvidia Geforce RTX 3090
RAM:128GiB
硬盘：HGST 5400转机械硬盘（PS4拆机）

耗时增加：1.5~3.0倍，smoother处理部分格外慢一些。如果上固态，速度影响将会减少很多，但不推荐，读写量非常大。

### 依赖调整

调整了一些依赖版本，并增加了requirements.txt方便在各种环境和IDE跑

### bug修复？

修了一个可能因为模型导致的文本生成视频时出现黑帧导致视频完全没法看的问题？

---

## 介绍

DiffSynth是一个新的扩散引擎。我们重构了包括文本编码器、UNet、VAE等在内的架构，同时保持与开源社区模型的兼容性，并提高了计算性能。此版本目前处于初始阶段，支持SD和SDXL架构。未来，我们计划在这个新代码库上开发更多有趣的功能。

## 安装

创建Python环境：

```python
conda env create -f environment.yml
```

我们发现`conda`有时无法正确安装`cupy`，请手动安装。详情请参阅[此文档](https://docs.cupy.dev/en/stable/install.html)。

进入Python环境：

```python
conda activate DiffSynthStudio
```

## 使用（在WebUI中）

```python
python -m streamlit run DiffSynth_Studio.py
```
https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/93085557-73f3-4eee-a205-9829591ef954

## 使用（Python代码方式）

### 示例1：Stable Diffusion

我们可以生成非常高分辨率的图像。更多详情请参阅`examples/sd_text_to_image.py`。

|512*512|1024*1024|2048*2048|4096*4096|
|-|-|-|-|
|![512](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/55f679e9-7445-4605-9315-302e93d11370)|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/6fc84611-8da6-4a1f-8fee-9a34eba3b4a5)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/9087a73c-9164-4c58-b2a0-effc694143fb)|![4096](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edee9e71-fc39-4d1c-9ca9-fa52002c67ac)|

### 示例2：Stable Diffusion XL

使用Stable Diffusion XL生成图像。更多详情请参阅`examples/sdxl_text_to_image.py`。

|1024*1024|2048*2048|
|-|-|
|![1024](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/67687748-e738-438c-aee5-96096f09ac90)|![2048](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/584186bc-9855-4140-878e-99541f9a757f)|

### 示例3：Stable Diffusion XL Turbo

使用Stable Diffusion XL Turbo生成图像。更多详情请参阅`examples/sdxl_turbo.py`，但我们强烈建议您在WebUI中使用。

|"black car"|"red car"|
|-|-|
|![black_car](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7fbfd803-68d4-44f3-8713-8c925fec47d0)|![black_car_to_red_car](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/aaf886e4-c33c-4fd8-98e2-29eef117ba00)|

### 示例4：卡通渲染（Diffutoon）

此示例基于[Diffutoon](https://arxiv.org/abs/2401.16224)实现。该方法适合渲染高分辨率视频和快速运动。您可以在配置中轻松修改参数。请参阅`examples/diffutoon_toon_shading.py`
。我们还提供[Colab示例](https://colab.research.google.com/github/Artiprocher/DiffSynth-Studio/blob/main/examples/Diffutoon.ipynb)。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd

### 示例5：带有编辑信号的卡通渲染（Diffutoon）

此示例基于[Diffutoon](https://arxiv.org/abs/2401.16224)，支持视频编辑信号。请参阅`examples\diffutoon_toon_shading_with_editing_signals.py`
。编辑功能也适用于[Colab示例](https://colab.research.google.com/github/Artiprocher/DiffSynth-Studio/blob/main/examples/Diffutoon.ipynb)。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/20528af5-5100-474a-8cdc-440b9efdd86c

### 示例6：卡通渲染（原生Python代码）

此示例适用于开发者。如果您不想使用配置来管理参数，可以查看`examples/sd_toon_shading.py`了解如何在原生Python代码中使用。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/607c199b-6140-410b-a111-3e4ffb01142c

### 示例7：文本到视频

给定提示，DiffSynth Studio可以使用Stable Diffusion模型和AnimateDiff模型生成视频。我们可以打破帧数的限制！请参阅`examples/sd_text_to_video.py`。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/8f556355-4079-4445-9b48-e9da77699437

### 示例8：视频风格化

我们提供一个视频风格化的示例。在这个管道中，渲染的视频与原始视频完全不同，因此我们需要一个强大的去闪烁算法。我们使用FastBlend实现去闪烁模块。请参阅`examples/sd_video_rerender.py`了解更多详情。

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/59fb2f7b-8de0-4481-b79f-0c3a7361a1ea

### 示例9：提示处理

如果您不是英语母语者，我们为您提供翻译服务。[翻译模型](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) 将翻译提示以获得更好的视觉质量。

Prompt: "一个漂亮的女孩". [翻译模型](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) 将翻译提示以获得更好的视觉质量。

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/ebb25ca8-7ce1-4d9e-8081-59a867c70c4d)|![1_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a7e79853-3c1a-471a-9c58-c209ec4b76dd)|![2_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/a292b959-a121-481f-b79c-61cc3346f810)|![3_](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/1c19b54e-5a6f-4d48-960b-a7b2b149bb4c)|

Prompt: "一个漂亮的女孩". [翻译模型](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) 将翻译提示以获得更好的视觉质量。

|seed=0|seed=1|seed=2|seed=3|
|-|-|-|-|
|![0](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/778b1bd9-44e0-46ac-a99c-712b3fc9aaa4)|![1](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/c03479b8-2082-4c6e-8e1c-3582b98686f6)|![2](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/edb33d21-3288-4a55-96ca-a4bfe1b50b00)|![3](https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/7848cfc1-cad5-4848-8373-41d24e98e584)|
