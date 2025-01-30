import torch
from model_loader import load_model, optimize_model_for_inference
from generator import generate_response
from config import Config
import logging
from pathlib import Path

def setup_logging(config: Config):
    Path(config.log_file.parent).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=config.log_file,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def main():
    config = Config()
    setup_logging(config)
    
    logging.info("加载模型和分词器...")
    try:
        model, tokenizer = load_model(config)
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    # 如果需要优化模型，可以取消注释以下代码
    logging.info("优化中")
    model = optimize_model_for_inference(model)
    
    logging.info("聊天机器人已加载。输入 'exit' 以退出。")
    while True:
        try:
            input_text = input("User: ")
            if input_text.lower() == 'exit':
                logging.info("聊天机器人已下线。")
                break
            response = generate_response(model, tokenizer, input_text, config)
            print(f"Model: {response}")
            logging.info(f"用户输入: {input_text} - 模型响应: {response}")
        except KeyboardInterrupt:
            logging.info("聊天机器人已手动中断。")
            break
        except Exception as e:
            logging.error(f"生成响应时发生错误: {e}")
            print("抱歉，发生了一些错误。请重试。")

if __name__ == "__main__":
    main()