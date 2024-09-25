import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator

from onediffx.deep_cache import StableDiffusionXLPipeline as StableDiffusionXLPipelineFaster
from onediffx import compile_pipe, load_pipe

def load_pipeline() -> StableDiffusionXLPipelineFaster:
    pipeline = StableDiffusionXLPipelineFaster.from_pretrained(
        "./models/newdream-sdxl-20",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    load_pipe(pipeline, dir="models/newdream-sdxl-20-optimized")
    pipeline(prompt="", cache_interval=2, cache_layer_id=0, cache_block_id=0)

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipelineFaster) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        cache_interval=2, cache_layer_id=0, cache_block_id=0,
        num_inference_steps=20,
    ).images[0]
