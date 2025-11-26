import argparse
import pathlib
import numpy as np
import onnx
import onnxsim
import torch
import os
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel, PreTrainedTokenizer, CLIPTextModelWithProjection
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from diffusers import DPMSolverMultistepScheduler
"""
test env:
    protobuf:3.20.3
    onnx:1.16.0
    onnxsim:0.4.36
    torch:2.1.2+cu121
    transformers:4.45.0
"""


def extract_by_hand(input_model):
    """从ONNX图中移除时间投影和嵌入节点，将其替换为外部输入"""
    input_graph = input_model
    to_remove_node = []
    for node in input_graph.node:
        if (
            node.name.startswith("/time_proj")
            or node.name.startswith("/time_embedding")
            or node.name
            in [
                "/down_blocks.0/resnets.0/act_1/Sigmoid",
                "/down_blocks.0/resnets.0/act_1/Mul",
            ]
        ):
            to_remove_node.append(node)
    for node in to_remove_node:
        input_graph.node.remove(node)
    
    # 移除原始的"t"输入
    to_remove_input = []
    for input in input_graph.input:
        if input.name in ["t"]:
            to_remove_input.append(input)
    for input in to_remove_input:
        input_graph.input.remove(input)
    
    # 添加新的时间嵌入输入
    new_input = []
    for value_info in input_graph.value_info:
        if value_info.name == "/down_blocks.0/resnets.0/act_1/Mul_output_0":
            new_input.append(value_info)
    input_graph.input.extend(new_input)

def extract_vae(args):
    # 使用与UNet导出相同的模型加载方式
    pipe = StableDiffusionPipeline.from_single_file(
        "./s/darkSushiMixMix_225D.safetensors",
        torch_dtype=torch.float32,
    )
    vae = pipe.vae
    vae.eval()

    # 解析输入尺寸
    isize = args.isize
    vae_encoder_img_h, vae_encoder_img_w = list(map(int, isize.split("x")))
    vae_decoder_img_h = vae_encoder_img_h // 8
    vae_decoder_img_w = vae_encoder_img_w // 8

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 导出VAE Decoder
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, post_quant_conv, decoder):
            super().__init__()
            self.post_quant_conv = post_quant_conv
            self.decoder = decoder

        def forward(self, latent):

            z = self.post_quant_conv(latent)
            return self.decoder(z)

    dummy_input = torch.randn(
        1, 4, vae_decoder_img_h, vae_decoder_img_w,
        dtype=torch.float32
    )

    torch.onnx.export(
        VAEDecoderWrapper(vae.post_quant_conv, vae.decoder),
        dummy_input,
        os.path.join(args.output_path, "vae_decoder.onnx"),
        opset_version=17,
        do_constant_folding=True,
        input_names=["latent"],
        output_names=["sample"],
    )

    # 简化ONNX模型
    vae_decoder_onnx = onnx.load(os.path.join(args.output_path, "vae_decoder.onnx"))
    simplified_model, check = onnxsim.simplify(vae_decoder_onnx)
    assert check, "Simplification failed"
    onnx.save(simplified_model, os.path.join(args.output_path, "vae_decoder_sim.onnx"))


def extract_unet_and_time_embedding(args):
    """导出UNet到ONNX并生成DPM++ 40步时间嵌入"""
    input_path = args.input_path
    output_path = args.output_path
    is_img2img = args.img2img
    isize = args.isize
    vae_encoder_img_h, vae_encoder_img_w = list(map(int, isize.split("x")))
    unet_feat_h = vae_encoder_img_h // 8
    unet_feat_w = vae_encoder_img_w // 8

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 加载基础模型
    #pipe = AutoPipelineForText2Image.from_pretrained(
    #    input_path, torch_dtype=torch.float32, variant="fp16"
    #)

    # 加载safetensors权重

    pipe = StableDiffusionPipeline.from_single_file(
        input_path,
        torch_dtype=torch.float32,
    )

    # 配置DPM++ 2M Karras采样器（40步）
    pipe.scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True,
        final_sigmas_type="sigma_min"
    )
    
    # 加载并融合LoRA权重
    #pipe.load_lora_weights(
    #     "./s/guofeng.safetensors",alpha=0.7
    #)
    #pipe.fuse_lora(lora_scale=0.7)
    
    # 生成测试图片验证LoRA效果
    #test_image = pipe(prompt=args.prompt, num_inference_steps=20, guidance_scale=7,    height=vae_encoder_img_h, 
    #width=vae_encoder_img_w).images[0]
    #test_image.save(pathlib.Path(output_path) / "test.jpg")

    
    """导出UNet到ONNX"""
    pipe.unet.eval()

    class UNETWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample=None, t=None, encoder_hidden_states=None):
            return self.unet.forward(sample, t, encoder_hidden_states)

    example_input = {
        "sample": torch.rand([1, 4, unet_feat_h, unet_feat_w], dtype=torch.float32),
        "t": torch.from_numpy(np.array([1], dtype=np.int64)),
        "encoder_hidden_states": torch.rand([1, 77, 768], dtype=torch.float32),
    }

    unet_path = pathlib.Path(output_path) / "unet"
    if not unet_path.exists():
        unet_path.mkdir()
    
    # 导出ONNX模型
    torch.onnx.export(
        UNETWrapper(pipe.unet),
        tuple(example_input.values()),
        str(unet_path / "unet.onnx"),
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        input_names=list(example_input.keys()),
    )
    
    # 简化ONNX模型
    unet = onnx.load(str(unet_path / "unet.onnx"))
    unet_sim, check = onnxsim.simplify(unet)
    assert check, "Simplified ONNX model could not be validated"

    # 移除时间嵌入相关节点，使其成为外部输入
    extract_by_hand(unet_sim.graph)
    onnx.save(
        unet_sim,
        str(pathlib.Path(output_path) / "unet_sim_cut.onnx"),
        save_as_external_data=True,
    )
    
    """生成DPM++ 40步时间嵌入"""
    num_inference_steps = 20  # DPM++ 40步采样
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = pipe.scheduler.timesteps.cpu().numpy()
    print(f"DPM++ 40步采样时间步: {timesteps}")
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=True
    )
    scheduler.set_timesteps(num_inference_steps)
    # 预计算时间嵌入
    time_embed_dim = pipe.unet.time_embedding.linear_1.out_features  # 通常为1280
    time_input = np.zeros([len(timesteps), time_embed_dim], dtype=np.float32)
    
    for i, t in enumerate(timesteps):
        # 转换时间步为模型需要的格式
        t = t * scheduler.init_noise_sigma  # 增加尺度调整
        
        tt = torch.tensor([t], dtype=torch.float32)
        
        # 计算时间投影和嵌入
        sample = pipe.unet.time_proj(tt)
        res = pipe.unet.time_embedding(sample)
        res = torch.nn.functional.silu(res)  # 应用SiLU激活函数
        
        # 保存到数组
        time_input[i, :] = res.detach().numpy()[0]
    
    # 保存时间嵌入到npy文件
    time_input_path = pathlib.Path(output_path) / "time_input_dpmpp_20steps.npy"
    np.save(str(time_input_path), time_input)
    print(f"时间嵌入已保存至: {time_input_path}")
    print(f"时间嵌入形状: {time_input.shape}")  # 应输出 (40, 1280)




def extract_text_encoder(args):
    """从safetensors文件导出文本编码器到ONNX"""
    # 复用与UNet相同的加载方式
    pipe = StableDiffusionPipeline.from_single_file(
        "./s/darkSushiMixMix_225D.safetensors",
        torch_dtype=torch.float32,
    )
    text_encoder = pipe.text_encoder
    text_encoder.eval()

    # 参数设置
    max_length = 77
    os.makedirs(args.output_path, exist_ok=True)

    # 改进的虚拟输入生成
    dummy_input = torch.zeros(
        (1, max_length), 
        dtype=torch.int64,  # 保持与tokenizer输出类型一致
        device=text_encoder.device
    )

    # 动态维度设置（支持动态batch）
    dynamic_axes = {
        "input_ids": {0: "batch"},
        "last_hidden_state": {0: "batch"}
    }

    # 导出完整的text_encoder（包含最终投影层）
    torch.onnx.export(
        text_encoder,
        dummy_input,
        str(pathlib.Path(args.output_path) / "text_encoder.onnx"),
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        # 显式指定hidden_states和pooler_output的忽略
        output_files=["last_hidden_state"],
    )

    # 模型简化与优化
    text_encoder_onnx = onnx.load(str(pathlib.Path(args.output_path) / "text_encoder.onnx"))
    simplified_model, check = onnxsim.simplify(
        text_encoder_onnx,
        overwrite_input_shapes={"input_ids": [1, max_length]},
        perform_optimization=True
    )
    assert check, "ONNX模型简化校验失败"
    
    # 移除冗余节点（适配新版导出结构）
    def extract_text_encoder_by_hand(graph):
        removed_nodes = ["/text_model/Add_1", "/text_model/Reshape_5"]
        graph.node[:] = [n for n in graph.node if n.name not in removed_nodes]
        
        # 更新输出维度信息
        for output in graph.output:
            if output.name == "last_hidden_state":
                output.type.tensor_type.shape.dim[0].dim_param = "batch"
                output.type.tensor_type.shape.dim[1].dim_value = max_length
                output.type.tensor_type.shape.dim[2].dim_value = 768

    extract_text_encoder_by_hand(simplified_model.graph)
    onnx.save(simplified_model, str(pathlib.Path(args.output_path) / "text_encoder_sim.onnx"))


if __name__ == "__main__":
    """
    用法示例:
        python sd15_export_unet_with_dpmpp_time_embedding.py \
            --input_path ../hugging_face/models/dreamshaper-7 \
            --input_lora_path ../hugging_face/models/lcm-lora-sdv1-5 \
            --output_path ./output_onnx_256x256 \
            --isize 256x256
    """
    parser = argparse.ArgumentParser(description="导出UNet到ONNX并生成DPM++ 时间嵌入")
    parser.add_argument("--input_path", required=True, help="基础模型路径")
    parser.add_argument("--prompt", default="480x320", required=True, help="原始模型测试prompt")
    parser.add_argument("--output_path", required=True, help="输出路径")
    parser.add_argument("--img2img", action="store_true", help="支持image-to-image模式")
    parser.add_argument("--isize", default="480x320", help="vae encoder输入图像尺寸")

    args = parser.parse_args()
    extract_unet_and_time_embedding(args)
    extract_vae(args)
    #extract_text_encoder(args)
