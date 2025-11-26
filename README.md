# sd15-to-ax8850-deploy
This repository demonstrates how to convert and deploy a Stable Diffusion 1.5 model to an AX8850 device.

本仓库用于演示如何将 Stable Diffusion 1.5 模型转换并部署到 AX8850 设备上。
示例流程将以从 C 站随机关联的 SD1.5 模型为例，完整展示以下内容：

从 C 站下载并准备 SD1.5 权重
模型格式转换与剪裁（适配 AX8850 推理框架）
在 AX8850 上的部署配置与推理脚本
简单的图像生成测试与性能验证
目标是为有 AX8850 硬件环境的用户提供一个可复现、可改造的参考项目，方便将自己的 SD1.5 模型迁移到 AX8850 上运行。
