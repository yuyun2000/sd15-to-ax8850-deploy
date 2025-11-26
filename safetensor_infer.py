import argparse
import pathlib
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler

def get_scheduler(scheduler_type, original_scheduler):
    """根据采样器类型配置调度器"""
    if scheduler_type == "dpm":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
            final_sigmas_type="sigma_min"
        )
    elif scheduler_type == "unipc":
        # 从原始调度器继承基本配置，避免参数不匹配
        return UniPCMultistepScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                use_karras_sigmas=True
            )
    else:
        raise ValueError(f"不支持的采样器类型: {scheduler_type}")

def test_model_with_lora(args):
    """加载模型和LoRA进行推理测试"""
    print("正在加载基础模型...")
    # 加载safetensors权重
    pipe = StableDiffusionPipeline.from_single_file(
        args.input_path,
        torch_dtype=torch.float32,
    )
    
    print(f"配置采样器: {args.scheduler}...")
    # 保存原始调度器配置
    original_scheduler = pipe.scheduler
    # 配置选定的采样器
    pipe.scheduler = get_scheduler(args.scheduler, original_scheduler)
    
    # 如果提供了LoRA路径，则加载LoRA
    if args.lora_path:
        print(f"正在加载LoRA: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
        print(f"LoRA权重设置为: {args.lora_scale}")
    
    # 移动到GPU（如果可用）
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("模型已移动到GPU")
    
    # 创建输出目录
    output_dir = pathlib.Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印生成参数
    print(f"采样器: {args.scheduler.upper()}")
    print(f"正面提示词: {args.prompt}")
    print(f"负面提示词: {args.negative_prompt}")
    print(f"CFG参数: {args.guidance_scale}")
    print(f"推理步数: {args.steps}")
    print(f"图片尺寸: {args.width}x{args.height}")
    
    # 对于UniPC + 低步数，调整CFG值
    guidance_scale = args.guidance_scale

    
    # 测试不同的LoRA权重（如果加载了LoRA）
    if args.lora_path:
        print("测试不同LoRA权重...")
        test_scales = [0.6, 0.7, 0.8] if args.test_multiple_scales else [args.lora_scale]
        for scale in test_scales:
            print(f"生成图片 - LoRA权重: {scale}")
            # 使用cross_attention_kwargs来控制LoRA权重
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=guidance_scale,
                width=args.width,
                height=args.height,
                cross_attention_kwargs={"scale": scale},
                generator=torch.Generator().manual_seed(42) if args.scheduler == "unipc" else None  # UniPC添加固定种子
            ).images[0]
            
            # 保存图片，文件名包含采样器信息
            image_path = output_dir / f"test_{args.scheduler}_lora_{scale}_steps_{args.steps}.jpg"
            image.save(image_path)
            print(f"图片已保存: {image_path}")
    else:
        print("生成图片（无LoRA）...")
        # 不使用LoRA的情况
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=guidance_scale,
            width=args.width,
            height=args.height,
            generator=torch.Generator().manual_seed(42) if args.scheduler == "unipc" else None  # UniPC添加固定种子
        ).images[0]
        
        # 保存图片
        image_path = output_dir / f"test_{args.scheduler}_no_lora_steps_{args.steps}.jpg"
        image.save(image_path)
        print(f"图片已保存: {image_path}")
    
    print("推理测试完成！")

def test_model_with_fused_lora(args):
    """使用融合LoRA的方式进行测试"""
    print("正在加载基础模型...")
    pipe = StableDiffusionPipeline.from_single_file(
        args.input_path,
        torch_dtype=torch.float32,
    )
    
    print(f"配置采样器: {args.scheduler}...")
    # 保存原始调度器配置
    original_scheduler = pipe.scheduler
    # 配置选定的采样器
    pipe.scheduler = get_scheduler(args.scheduler, original_scheduler)
    
    if args.lora_path:
        print(f"正在加载并融合LoRA: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
        pipe.fuse_lora(lora_scale=args.lora_scale)
        print(f"LoRA已融合，权重: {args.lora_scale}")
    
    # 移动到GPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("模型已移动到GPU")
    
    # 创建输出目录
    output_dir = pathlib.Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印生成参数
    print(f"采样器: {args.scheduler.upper()}")
    print(f"正面提示词: {args.prompt}")
    print(f"负面提示词: {args.negative_prompt}")
    print(f"CFG参数: {args.guidance_scale}")
    print(f"推理步数: {args.steps}")
    print(f"图片尺寸: {args.width}x{args.height}")
    
    # 对于UniPC + 低步数，调整CFG值
    guidance_scale = args.guidance_scale
    if args.scheduler == "unipc" and args.steps <= 10:
        guidance_scale = min(args.guidance_scale * 1.2, 12.0)
        print(f"UniPC低步数模式，CFG调整为: {guidance_scale}")
    
    print("生成测试图片...")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=guidance_scale,
        width=args.width,
        height=args.height,
        generator=torch.Generator().manual_seed(42) if args.scheduler == "unipc" else None
    ).images[0]
    
    # 保存图片
    suffix = f"_fused_lora_{args.lora_scale}" if args.lora_path else "_no_lora"
    image_path = output_dir / f"test_{args.scheduler}{suffix}_steps_{args.steps}.jpg"
    image.save(image_path)
    print(f"图片已保存: {image_path}")
    print("推理测试完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试Stable Diffusion模型和LoRA效果")
    
    # 必需参数
    parser.add_argument("--input_path", required=True, help="基础模型safetensors文件路径")
    parser.add_argument("--output_path", required=True, help="测试图片输出路径")
    parser.add_argument("--prompt", required=True, help="生成图片的提示词")
    
    # LoRA相关参数
    parser.add_argument("--lora_path", help="LoRA权重文件路径")
    parser.add_argument("--lora_scale", type=float, default=0.7, help="LoRA权重强度 (推荐0.6-0.8)")
    
    # 生成参数
    parser.add_argument("--negative_prompt", required=True, help="负面提示词")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="CFG引导强度")
    parser.add_argument("--width", type=int, default=320, help="图片宽度")
    parser.add_argument("--height", type=int, default=480, help="图片高度")
    
    # 采样器选择参数
    parser.add_argument("--scheduler", choices=["dpm", "unipc"], default="dpm", 
                       help="选择采样器: dmp (DPM++ 2M Karras) 或 unipc (UniPC)")
    
    # 测试选项
    parser.add_argument("--test_multiple_scales", action="store_true", help="测试多个LoRA权重值")
    parser.add_argument("--use_fused_lora", action="store_true", help="使用融合LoRA模式")
    
    args = parser.parse_args()
    
    # 根据选择的模式运行测试
    if args.use_fused_lora:
        test_model_with_fused_lora(args)
    else:
        test_model_with_lora(args)