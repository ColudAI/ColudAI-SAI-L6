import torch
from model_loader import load_model, optimize_model_for_inference
from generator import generate_response
from config import CONFIG, LOGGER  # 使用全局配置和日志对象
from pathlib import Path
import sys

def setup_environment():
    """初始化运行环境"""
    # 创建必要目录
    Path(CONFIG.LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(CONFIG.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # 设置CUDA线程数（提升推理性能）
    if torch.cuda.is_available():
        torch.set_num_threads(4)

def validate_input(text: str) -> bool:
    """验证用户输入有效性"""
    text = text.strip()
    if not text:
        LOGGER.warning("收到空输入")
        return False
    if len(text) > CONFIG.MAX_INPUT_LENGTH:
        LOGGER.warning(f"输入过长: {len(text)} > {CONFIG.MAX_INPUT_LENGTH}")
        return False
    return True

def main():
    """主程序入口"""
    setup_environment()
    
    LOGGER.info("=== 系统启动 ===")
    LOGGER.info(f"硬件配置: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # 加载模型和分词器
        LOGGER.info("正在加载模型...")
        model, tokenizer = load_model(
            model_path=CONFIG.MODEL_PATH,
            device=CONFIG.DEVICE,
            use_quantization=CONFIG.USE_QUANTIZATION
        )
        
        # 模型优化（根据配置决定）
        if CONFIG.OPTIMIZE_FOR_INFERENCE:
            LOGGER.info("应用推理优化...")
            model = optimize_model_for_inference(
                model,
                use_fp16=CONFIG.USE_FP16,
                use_tf32=CONFIG.USE_TF32
            )
            
        LOGGER.info("模型加载完成")
        print("\n" + "="*40)
        print(f"欢迎使用 {CONFIG.BOT_NAME}！输入 'exit' 退出")
        print("="*40 + "\n")

        # 交互循环
        while True:
            try:
                input_text = input("User: ").strip()
                if input_text.lower() in ('exit', 'quit'):
                    LOGGER.info("用户请求退出")
                    break
                
                # 输入验证
                if not validate_input(input_text):
                    print("提示：请输入有效内容（长度<500字符）")
                    continue
                
                # 生成响应
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=input_text,
                    max_length=CONFIG.GENERATION_MAX_LENGTH,
                    temperature=CONFIG.GENERATION_TEMPERATURE,
                    top_p=CONFIG.GENERATION_TOP_P,
                    device=CONFIG.DEVICE
                )
                
                # 输出处理
                cleaned_response = response.replace(CONFIG.EOS_TOKEN, "").strip()
                print(f"\n{CONFIG.BOT_NAME}: {cleaned_response}\n")
                LOGGER.info(f"交互日志 | 输入: {input_text} | 输出: {cleaned_response}")
                
            except KeyboardInterrupt:
                LOGGER.warning("用户中断操作")
                print("\n提示：输入 'exit' 可退出程序")
                continue
                
    except Exception as e:
        LOGGER.critical(f"系统崩溃: {str(e)}", exc_info=True)
        sys.exit(1)
        
    finally:
        LOGGER.info("=== 系统关闭 ===")
        if 'model' in locals():
            del model  # 显式释放显存
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()