# -*- coding: utf-8 -*-

from ultralytics import YOLO
import sys
import os

# 获取命令行参数
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (default: 640)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--source', type=str, required=True, help='Source of images, can be file, directory or video')
    parser.add_argument('--device', type=str, default='0', help='Device to use for inference (e.g., 0 for GPU, cpu for CPU)')
    parser.add_argument('--save_txt', action='store_true', help='Save results to a .txt file')
    parser.add_argument('--save_img', action='store_true', help='Save results to images')
    return parser.parse_args()

def main():
    # 解析命令行参数
    opt = parse_args()

    # 加载 YOLO 模型
    model = YOLO(opt.weights)

    # 进行推理
    results = model.predict(
        source=opt.source,   # 输入源：图片、视频或文件夹路径
        imgsz=opt.imgsz,     # 输入图像大小
        conf=opt.conf,       # 置信度阈值
        device=opt.device    # 使用的设备（GPU或CPU）
    )

    # 如果选择保存图像
    if opt.save_img:
        results.save()  # 保存推理后的图像（默认保存在 runs/detect/ 目录）

    # 如果选择保存结果为文本文件
    if opt.save_txt:
        results.save_txt()  # 保存推理结果的文本文件（每个检测框的坐标和类别）

    # 打印推理结果
    print(f"Results saved to {os.path.abspath(results.save_dir)}")

if __name__ == '__main__':
    main()
