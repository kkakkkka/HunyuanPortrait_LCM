import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from src.models.condition.unet_3d_svd_condition_ip import UNetSpatioTemporalConditionModel 
from torch.utils.data import DataLoader

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        self.ddim_sigmas = np.zeros_like(self.ddim_timesteps)

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = self.ddim_alpha_cumprods_prev[timestep_index]
        dir_xt = (1.0 - alpha_cumprod_prev - self.ddim_sigmas[timestep_index] ** 2).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

def main(args):
    accelerator = Accelerator(mixed_precision="fp16")
    
    teacher_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_path, subfolder="unet"
    )
    teacher_unet.requires_grad_(False)
    teacher_unet.to(accelerator.device, dtype=torch.float16)

    student_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_path, subfolder="unet"
    )
    student_unet.requires_grad_(False)
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
    )
    student_unet = get_peft_model(student_unet, lora_config)
    student_unet.train()
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    alpha_cumprods = noise_scheduler.alphas_cumprod.numpy()
    solver = DDIMSolver(alpha_cumprods, timesteps=noise_scheduler.config.num_train_timesteps)

    optimizer = torch.optim.AdamW(
        student_unet.parameters(), 
        lr=args.learning_rate, 
        weight_decay=1e-4
    )

    train_dataset = [] 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    student_unet, optimizer, train_dataloader = accelerator.prepare(
        student_unet, optimizer, train_dataloader
    )

    for epoch in range(args.num_epochs):
        for batch in tqdm(train_dataloader, disable=not accelerator.is_local_main_process):
            latents = batch["latents"].to(accelerator.device, dtype=torch.float16)
            encoder_hidden_states = batch["encoder_hidden_states"].to(accelerator.device, dtype=torch.float16)
            added_time_ids = batch["added_time_ids"].to(accelerator.device, dtype=torch.float16)

            bsz = latents.shape[0]
            
            topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=accelerator.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            
            c_skip_routing = args.num_ddim_timesteps // args.lcm_steps
            next_index = torch.clamp(index - c_skip_routing, min=0)
            next_timesteps = solver.ddim_timesteps[next_index]

            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, start_timesteps)

            student_pred = student_unet(
                noisy_latents, 
                start_timesteps, 
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids
            ).sample

            with torch.no_grad():
                teacher_pred = teacher_unet(
                    noisy_latents, 
                    start_timesteps, 
                    encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=added_time_ids
                ).sample
                
                pred_x0 = (noisy_latents - solver.ddim_sigmas[index].view(-1,1,1,1,1) * teacher_pred) / solver.ddim_alpha_cumprods_prev[index].sqrt().view(-1,1,1,1,1)
                target_latents = solver.ddim_step(pred_x0, teacher_pred, next_index)

            loss = F.huber_loss(student_pred.float(), target_latents.float(), delta=args.huber_c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        if accelerator.is_local_main_process and (epoch + 1) % args.save_every == 0:
            unwrapped_model = accelerator.unwrap_model(student_unet)
            unwrapped_model.save_pretrained(os.path.join(args.output_dir, f"lcm-lora-epoch-{epoch+1}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lcm_lora_output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_ddim_timesteps", type=int, default=50)
    parser.add_argument("--lcm_steps", type=int, default=4)
    parser.add_argument("--huber_c", type=float, default=0.001)
    args = parser.parse_args()
    main(args)
