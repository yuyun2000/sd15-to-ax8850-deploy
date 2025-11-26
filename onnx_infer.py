from typing import List, Union
import numpy as np
import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
import os
import time
import argparse
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline

def get_args():
    parser = argparse.ArgumentParser(
        prog="StableDiffusion",
        description="Generate picture with precomputed time embeddings (ONNX with merged LoRA)"
    )
    parser.add_argument("--prompt", type=str, required=False, default="1girl, upper body, (huge Laughing),sweety,sun glare, bokeh, depth of field, blurry background, light particles, strong wind,head tilt,simple background, red background,<lora:film:0.4>")
    parser.add_argument("--text_model_dir", type=str, required=False, default="./out_onnx/", help="Path to text encoder and tokenizer files")
    parser.add_argument("--unet_model", type=str, required=False, default="./out_onnx/unet_sim_cut.onnx", help="Path to unet ONNX model (with merged LoRA)")
    parser.add_argument("--vae_decoder_model", type=str, required=False, default="./out_onnx/vae_decoder_sim.onnx", help="Path to vae decoder ONNX model")
    parser.add_argument("--time_input", type=str, required=False, default="./out_onnx/time_input_dpmpp_20steps.npy", help="Path to precomputed time embeddings")
    parser.add_argument("--num_inference_steps", type=int, required=False, default=20, help="Number of inference steps (must match time_input)")
    parser.add_argument("--guidance_scale", type=float, required=False, default=7, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--seed", type=int, required=False, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, required=False, default="./txt2img_output_onnx.png", help="Path to the output image file")
    return parser.parse_args()


def get_prompt_embeddings(prompt, negative_prompt, tokenizer, text_encoder):
    """获取提示词和负面提示词的嵌入（分别获取，而不是拼接）"""
    # 处理提示词
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    
    # 处理负面提示词
    negative_text_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    
    # 获取文本嵌入
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids)[0]  # 形状: [1, 77, 768]
        negative_prompt_embeds = text_encoder(negative_text_inputs.input_ids)[0]  # 形状: [1, 77, 768]
    
    return negative_prompt_embeds, prompt_embeds


def main():
    args = get_args()
    
    # 设置随机种子以确保可重现性
    seed = args.seed if args.seed is not None else int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")
    
    # 默认负面提示词
    negative_prompt = "easynegative,ng_deepnegative_v1_75t,(worst quality:2),(low quality:2),(normal quality:2),lowres,bad anatomy,bad hands,normal quality,((monochrome)),((grayscale)),((watermark)),"
    
    # 加载tokenizer和text encoder
    print("Loading tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("./tokenizer")
    pipe = StableDiffusionPipeline.from_single_file(
        "./s/xxmix9realistic_v40.safetensors",
        torch_dtype=torch.float32,
    )
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    
    # 获取提示词嵌入（分开获取负面和正面）
    print("Generating prompt embeddings...")
    start = time.time()
    negative_embeds, positive_embeds = get_prompt_embeddings(args.prompt, negative_prompt, tokenizer, text_encoder)
    
    # 转换为numpy数组
    negative_embeds_np = negative_embeds.numpy()  # 形状: [1, 77, 768]
    positive_embeds_np = positive_embeds.numpy()  # 形状: [1, 77, 768]
    
    print(f"Negative embeddings shape: {negative_embeds_np.shape}")
    print(f"Positive embeddings shape: {positive_embeds_np.shape}")
    print(f"Prompt embeddings generated in {(time.time() - start):.2f} seconds")
    
    # 初始化DPM++调度器
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
    scheduler.set_timesteps(args.num_inference_steps)
    timesteps = scheduler.timesteps
    print(f"DPM++ timesteps: {timesteps.numpy()}")
    
    # 初始化潜在变量（批次大小为1，因为UNet不支持批次为2）
    unet_feat_h, unet_feat_w = 60, 40  # 根据您的UNet输入形状
    latents_shape = [1, 4, unet_feat_h, unet_feat_w]
    latents = torch.randn(latents_shape, generator=torch.Generator().manual_seed(seed))
    
    # 使用调度器正确缩放初始噪声
    latents = latents * scheduler.init_noise_sigma
    latents_np = latents.numpy()
    
    # 加载预计算的时间嵌入
    print(f"Loading precomputed time embeddings from {args.time_input}...")
    time_input = np.load(args.time_input)
    
    # 验证时间嵌入形状
    if len(time_input) != args.num_inference_steps:
        raise ValueError(f"Time input length ({len(time_input)}) does not match num_inference_steps ({args.num_inference_steps})")
    print(f"Time embeddings shape: {time_input.shape}")  # 应该是 (40, 1280)
    
    # 加载ONNX模型
    print("Loading ONNX models...")
    start = time.time()
    
    # 设置ONNX推理会话选项
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 加载UNet模型（已合并LoRA，带预计算时间嵌入）
    unet_session = onnxruntime.InferenceSession(
        args.unet_model, 
        session_options,
        providers=["CPUExecutionProvider"]  # 如果有GPU可用，优先使用"CUDAExecutionProvider"
    )
    
    # 加载VAE解码器
    vae_decoder_session = onnxruntime.InferenceSession(
        args.vae_decoder_model,
        session_options,
        providers=["CPUExecutionProvider"]
    )
    
    print(f"Models loaded in {(time.time() - start):.2f} seconds")
    
    # 验证UNet输入
    unet_inputs = {input.name: input for input in unet_session.get_inputs()}
    required_inputs = {"sample", "encoder_hidden_states", "/down_blocks.0/resnets.0/act_1/Mul_output_0"}
    if not required_inputs.issubset(unet_inputs.keys()):
        missing = required_inputs - unet_inputs.keys()
        raise ValueError(f"UNet model is missing required inputs: {missing}")
    
    # 打印UNet输入信息用于调试
    print("UNet inputs:")
    for name, input in unet_inputs.items():
        print(f"  Name: {name}, Shape: {input.shape}, Type: {input.type}")
    
    # DPM++采样循环
    print("Starting DPM++ sampling...")
    start = time.time()
    
    for i, timestep in enumerate(timesteps):
        step_start = time.time()
        
        # 获取当前步骤的预计算时间嵌入
        current_time_embedding = time_input[i]
        
        # 分别为负面和正面提示词运行UNet推理
        
        # 1. 负面提示词推理
        unet_input_neg = {
            "sample": latents_np,  # 当前潜在变量
            "encoder_hidden_states": negative_embeds_np,  # 负面提示词嵌入
            "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(current_time_embedding, axis=0)
        }
        noise_pred_neg = unet_session.run(None, unet_input_neg)[0]
        
        # 2. 正面提示词推理
        unet_input_pos = {
            "sample": latents_np,  # 当前潜在变量（与负面提示词相同）
            "encoder_hidden_states": positive_embeds_np,  # 正面提示词嵌入
            "/down_blocks.0/resnets.0/act_1/Mul_output_0": np.expand_dims(current_time_embedding, axis=0)
        }
        noise_pred_pos = unet_session.run(None, unet_input_pos)[0]
        
        # 3. 应用Classifier-Free Guidance
        if args.guidance_scale != 1.0:
            noise_pred = noise_pred_neg + args.guidance_scale * (noise_pred_pos - noise_pred_neg)
        else:
            noise_pred = noise_pred_pos
        
        # 使用调度器计算下一步潜在变量（修复：移除prev_timestep参数）
        latents = torch.tensor(latents_np)
        noise_pred = torch.tensor(noise_pred)
        
        # 修复：获取当前和下一个sigma值（对于DPM++调度器，我们需要sigma值而非timestep）
        # 获取当前sigma
        sigma = scheduler.sigmas[i]
        
        # 获取下一个sigma（如果是最后一步，则为0）
        next_sigma = scheduler.sigmas[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0)
        
        # 使用当前sigma和下一个sigma更新潜在变量（适配新版Diffusers API）
        scheduler_output = scheduler.step(
            model_output=noise_pred,
            timestep=timestep,  # 使用当前timestep
            sample=latents
        )
    
        latents = scheduler_output.prev_sample
        latents_np = latents.numpy()
        
        print(f"Step {i+1}/{len(timesteps)} completed in {(time.time() - step_start):.2f} seconds")
    
    print(f"DPM++ sampling completed in {(time.time() - start):.2f} seconds")
    
    # VAE解码
    print("Decoding with VAE...")
    start = time.time()
    
    # 对潜在变量进行缩放（使用正确的缩放因子）
    latents_np = latents_np / 0.18215
    
    # 运行VAE解码器
    image = vae_decoder_session.run(None, {"latent": latents_np})[0]
    
    print(f"VAE decoding completed in {(time.time() - start):.2f} seconds")
    
    # 后处理图像
    print("Postprocessing image...")
    # 转置维度 (batch, channels, height, width) -> (batch, height, width, channels)
    image = np.transpose(image, (0, 2, 3, 1))[0]  # 取第一个样本
    # 归一化到[0, 1]范围
    image = np.clip(image / 2 + 0.5, 0, 1)
    # 转换为uint8并保存
    image = (image * 255).astype(np.uint8)
    
    # 保存图像
    pil_image = Image.fromarray(image)
    pil_image.save(args.save_dir)
    print(f"Image saved to {args.save_dir}")


if __name__ == '__main__':
    main()
