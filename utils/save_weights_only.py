import torch
import argparse

import argparse  # 导入argparse模块，用于解析命令行参数

def get_opts():
    parser = argparse.ArgumentParser()  # 创建一个参数解析器

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='checkpoint path')  # 添加一个命令行参数ckpt_path，类型为字符串，必需，帮助信息为checkpoint路径

    return parser.parse_args()  # 解析命令行参数并返回结果

if __name__ == "__main__":  # 如果当前模块被直接执行
    args = get_opts()  # 解析命令行参数
    checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))  # 加载checkpoint，设备为cpu
    torch.save(checkpoint['state_dict'], args.ckpt_path.split('/')[-2]+'.ckpt')  # 保存state_dict到指定文件名中
    print('Done!')  # 表示完成
