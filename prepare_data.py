from typing import List, Union
import numpy as np
import os
import tarfile
import onnxruntime
import torch
import random
from diffusers import DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline

# 优化后的负面提示词库（已去重和结构化）
NEGATIVE_PROMPT_LIB = [
    "sketch, duplicate, ugly, huge eyes, text, logo, worst face",
    "(bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0)",
    "horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy",
    "(interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4)",
    "(deformed fingers:1.2), (long fingers:1.2), bad-artist-anime, bad-artist, bad hand",
    "extra legs, nipples, nsfw, monochrome, greyscale, topless male",
    "(low quality, worst quality:1.4), (FastNegativeEmbedding:0.9)",
    "EasyNegativeV2, (bad anatomy), (mutated limbs:1.2), (mutated hands:1.2)",
    "(text:1.5), simple background, paintings, sketches,(bad-hands-5:1)",
    "lowres, blurry, floating limbs, extra limb, malformed limbs, long neck",
    "cross-eyed, bad body, ugly, disgusting, bad feet, bad leg",
    "missing limb, disconnected limbs, extra legs, missing legs, extra foot",
    "bad eyes, acnes, skin blemishes, signature, watermark, username, duplicate",
    "(2girl), feminine, feminine posture, normal quality:1.31, worst quality:1.33",
    "FastNegativeEmbedding:0.9, easynegative, ng_deepnegative_v1_75t"
]

def get_random_negative_prompt() -> str:
    """从负面词库中随机选择2-4个元素组合生成"""
    selected = random.sample(NEGATIVE_PROMPT_LIB, k=random.randint(2,4))
    return ", ".join(selected).replace(",,", ",")

def get_prompt_embeddings(prompt: str, 
                         negative_prompt: str,
                         tokenizer: CLIPTokenizer,
                         text_encoder: CLIPTextModel) -> tuple:
    """生成符合实际分布的正负提示嵌入"""
    # 正提示处理
    pos_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    
    # 负提示处理
    neg_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        pos_embeds = text_encoder(pos_inputs.input_ids)[0].numpy()
        neg_embeds = text_encoder(neg_inputs.input_ids)[0].numpy()
    
    return neg_embeds, pos_embeds

def initialize_scheduler(num_steps: int):
    """初始化与推理代码完全同步的调度器配置"""
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True,
        final_sigmas_type="sigma_min"
    )
    # 关键修复：设置正确的时间步数并验证
    scheduler.set_timesteps(num_steps)
    assert len(scheduler.timesteps) == num_steps, f"步数不匹配，应为{num_steps}，实际{len(scheduler.timesteps)}"
    return scheduler
def generate_calibration_data():
    NUM_STEPS = 20
    LATENT_SHAPE = [1, 4, 60, 40]
    GUIDANCE_SCALE = 9.0
    # 公共组件初始化
    tokenizer = CLIPTokenizer.from_pretrained("./tokenizer")

    pipe = StableDiffusionPipeline.from_single_file(
        "./s/darkSushiMixMix_225D.safetensors",
        torch_dtype=torch.float32,
    )
    text_encoder = pipe.text_encoder
    text_encoder.eval()

    time_embeddings = np.load("out_onnx/time_input_dpmpp_20steps.npy")
    unet_session = onnxruntime.InferenceSession("out_onnx/unet_sim_cut.onnx")
    # 创建输出目录
    os.makedirs("calib_data_unet", exist_ok=True)
    os.makedirs("calib_data_vae", exist_ok=True)
    with tarfile.open("calib_data_unet/data.tar", "w") as calib_unet, \
         tarfile.open("calib_data_vae/data.tar", "w") as calib_vae:
        PROMPTS = [
            'a 30 yo woman,(hi-top fade:1.3),long hair,dark theme, soothing tones, muted colors, high contrast, (natural skin texture, hyperrealism, soft light, sharp),',
            'blonde, Curly hair,(hi-top fade:1.3), dark theme, soothing tones, muted colors, high contrast, (natural skin texture, hyperrealism, soft light, sharp),exposure blend, medium shot, bokeh, (hdr:1.4), high contrast, (cinematic, teal and orange:0.85), (muted colors, dim colors, soothing tones:1.3), low saturation, (hyperdetailed:1.2), (noir:0.4)',
            '(((gold, silver, glimmer)), faerie), limited palette, contrast, phenomenal aesthetic, best quality, sumptuous artwork',
            'a 20 yo woman, blonde, (hi-top fade:1.3), dark theme, soothing tones, muted colors, high contrast, (natural skin texture, hyperrealism, soft light, sharp)',
            'woman, flower dress, colorful, darl background,flower armor,green theme,exposure blend, medium shot, bokeh, (hdr:1.4), high contrast, (cinematic, teal and orange:0.85), (muted colors, dim colors, soothing tones:1.3), low saturation,',
            'A girl sitting on a giant ice cream, which is adorned with vibrant colors, delightful frosting, and rainbow sprinkles. She holds an oversized waffle cone,shimmering candies. Floating around her are some small balloons, each tied to a petite candy. The scene is filled with sweetness and joy, showcasing the girls happiness and the enchanting fusion of her imagination and a fantastical world,fantasy, high contrast, ink strokes, explosions, over exposure, purple and red tone impression , abstract, ((watercolor painting by John Berkey and Jeremy Mann )) brush strokes, negative space,',
            '1girl, upper body, (huge Laughing),sweety,sun glare, bokeh, depth of field, blurry background, light particles, strong wind,head tilt,simple background, red background,<lora:film:0.4>'
        ]
        
        for p_idx, prompt in enumerate(PROMPTS):
            # 关键修复：每个prompt使用独立的调度器
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
                final_sigmas_type="sigma_min"
            )
            scheduler.set_timesteps(NUM_STEPS)
            
            # 验证调度器状态
            assert len(scheduler.timesteps) == NUM_STEPS
            assert len(time_embeddings) == NUM_STEPS
            # 生成动态负提示
            neg_prompt = get_random_negative_prompt()
            neg_emb, pos_emb = get_prompt_embeddings(prompt, neg_prompt, tokenizer, text_encoder)
            
            # 初始化潜在变量
            generator = torch.Generator().manual_seed(42 + p_idx)
            latents = torch.randn(LATENT_SHAPE, generator=generator) * scheduler.init_noise_sigma
            latents_np = latents.numpy()
            # 分步处理
            for step in range(NUM_STEPS):
                current_timestep = scheduler.timesteps[step]
                time_emb = np.expand_dims(time_embeddings[step], axis=0)
                # UNET双路径采集
                for emb_type, emb in [("neg", neg_emb), ("pos", pos_emb)]:
                    unet_input = {
                        "sample": latents_np,
                        "encoder_hidden_states": emb,
                        "/down_blocks.0/resnets.0/act_1/Mul_output_0": time_emb
                    }
                    np.save(f"calib_data_unet/data_{p_idx}_{step}_{emb_type}.npy", unet_input)
                    calib_unet.add(f"calib_data_unet/data_{p_idx}_{step}_{emb_type}.npy")
                
                # 实际推理更新潜在变量
                noise_pred_neg = unet_session.run(None, {
                    "sample": latents_np,
                    "encoder_hidden_states": neg_emb,
                    "/down_blocks.0/resnets.0/act_1/Mul_output_0": time_emb
                })[0]
                
                noise_pred_pos = unet_session.run(None, {
                    "sample": latents_np,
                    "encoder_hidden_states": pos_emb,
                    "/down_blocks.0/resnets.0/act_1/Mul_output_0": time_emb
                })[0]
                
                noise_pred = noise_pred_neg + GUIDANCE_SCALE * (noise_pred_pos - noise_pred_neg)
                
                # 使用调度器当前时间步
                latents = scheduler.step(
                    model_output=torch.from_numpy(noise_pred),
                    timestep=current_timestep,
                    sample=torch.from_numpy(latents_np)
                ).prev_sample
                latents_np = latents.numpy()
            # VAE数据保存
            vae_input = latents_np / 0.18215
            np.save(f"calib_data_vae/data_{p_idx}.npy", {"latent": vae_input})
            calib_vae.add(f"calib_data_vae/data_{p_idx}.npy")
            

if __name__ == '__main__':
    generate_calibration_data()
