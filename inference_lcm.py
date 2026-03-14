import argparse
import torch
from peft import PeftModel
from diffusers import LCMScheduler
from diffusers.utils import load_image
from src.pipelines.hunyuan_svd_pipeline import HunyuanSVDPipeline

def main(args):
    pipeline = HunyuanSVDPipeline.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.float16
    ).to("cuda")

    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet, 
        args.lora_weights_path, 
        adapter_name="lcm-lora"
    )
    pipeline.unet.merge_and_unload()

    image = load_image(args.image_path)
    pose_image = load_image(args.pose_path)

    generator = torch.manual_seed(args.seed)

    output = pipeline(
        image=image,
        pose_image=pose_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        fps=args.fps,
        decode_chunk_size=args.decode_chunk_size,
    ).frames[0]

    from diffusers.utils import export_to_video
    export_to_video(output, args.output_path, fps=args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--lora_weights_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--pose_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="lcm_output.mp4")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--decode_chunk_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
