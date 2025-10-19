# # -*- coding: utf-8 -*-
#
# import sys
# import os
# import glob
# import numpy as np
# import torch
# from PIL import Image
# import argparse
# import pandas as pd
# import warnings
#
# # 确保 'core' 目录在 Python 路径中
# sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
#
# from core.raft import RAFT
# from core.utils.utils import InputPadder
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # ====== 可调参数（保持与你原脚本一致的默认） ======
# FPS = 60.0
# GSD_M_PER_PX = 0.0128     # 你的GSD
# GAPS = [3, 5, 7]             # 需要输出的帧间隔
#
# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     if img.ndim == 2:  # 灰度图转RGB
#         img = np.stack([img] * 3, axis=-1)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to(DEVICE)
#
# def compute_flow_statistics(flow, gap, fps=FPS, gsd=GSD_M_PER_PX):
#     """
#     flow: torch.Tensor, shape [1, 2, H, W], 单位=像素位移（两帧之间）
#     gap:  帧间隔（两图之间相隔的帧数），时间=gap/fps
#     返回：像素域统计 + 物理速度统计（m/s）
#     """
#     flow_np = flow[0].detach().cpu().numpy()
#     u_px, v_px = flow_np[0], flow_np[1]
#     mag_px = np.sqrt(u_px ** 2 + v_px ** 2)
#
#     # 像素→物理速度（m/s）：位移(px)*GSD(m/px) / (gap/fps) = px*GSD*fps/gap
#     scale = (fps / float(gap)) * gsd
#     u_ms = u_px * scale
#     v_ms = v_px * scale
#     mag_ms = mag_px * scale
#
#     stats = {
#         # 像素域（可选，便于检查）
#         'Mean_U_px': float(np.mean(u_px)),
#         'Mean_V_px': float(np.mean(v_px)),
#         'Mean_Mag_px': float(np.mean(mag_px)),
#
#         # 物理域（主要关注）
#         'Mean_U': float(np.mean(u_ms)),
#         'Mean_V': float(np.mean(v_ms)),
#         'Mean_Magnitude': float(np.mean(mag_ms)),
#         'Max_Magnitude': float(np.max(mag_ms)),
#     }
#     return stats
#
# def make_pairs(images, gap):
#     """
#     给定有序图片列表，按 gap 形成 (i, i+gap) 配对
#     """
#     n = len(images)
#     if n <= gap:
#         return []
#     return [(images[i], images[i + gap]) for i in range(0, n - gap)]
#
# def demo(args, output_excel, folder_index):
#     # 加载模型（保持你的写法）
#     model = torch.nn.DataParallel(RAFT(args))
#     model.load_state_dict(torch.load(args.model, map_location=DEVICE))
#     model = model.module.to(DEVICE).eval()
#
#     with torch.no_grad():
#         supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif')
#         images = sorted(sum([glob.glob(os.path.join(args.path, fmt)) for fmt in supported_formats], []))
#
#         if len(images) < 2:
#             print(f"[警告] 第 {folder_index} 个文件夹：至少需要两张图片进行光流计算，已跳过。")
#             return
#
#         # 针对每个 gap 分别计算，并写入一个 Excel 的不同 sheet
#         writer = pd.ExcelWriter(output_excel, engine='openpyxl')
#         all_rows = []  # 汇总到 all_gaps sheet
#
#         for gap in GAPS:
#             pairs = make_pairs(images, gap)
#             if not pairs:
#                 print(f"[提示] 第 {folder_index} 个文件夹：gap={gap} 可用配对为空（图片太少），跳过该gap。")
#                 continue
#
#             rows = []
#             for imfile1, imfile2 in pairs:
#                 image1, image2 = load_image(imfile1), load_image(imfile2)
#                 padder = InputPadder(image1.shape)
#                 image1, image2 = padder.pad(image1, image2)
#
#                 flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
#
#                 flow_stats = compute_flow_statistics(flow_up, gap=gap, fps=FPS, gsd=GSD_M_PER_PX)
#                 row = {
#                     'Image1': os.path.basename(imfile1),
#                     'Image2': os.path.basename(imfile2),
#                     'Gap': gap,
#                     **flow_stats
#                 }
#                 rows.append(row)
#                 all_rows.append(row)
#
#             if rows:
#                 df_gap = pd.DataFrame(rows)
#                 sheet_name = f"gap_{gap}"
#                 df_gap.to_excel(writer, index=False, sheet_name=sheet_name)
#
#                 # 简要统计打印
#                 print(f"[完成] Folder {folder_index} | gap={gap} | "
#                       f"Mean(Mean_Magnitude)={df_gap['Mean_Magnitude'].mean():.4f} m/s | "
#                       f"Mean(Max_Magnitude)={df_gap['Max_Magnitude'].mean():.4f} m/s")
#
#         # 写一个总汇总表
#         if all_rows:
#             df_all = pd.DataFrame(all_rows)
#             df_all.to_excel(writer, index=False, sheet_name="all_gaps")
#         writer.close()
#
#         if all_rows:
#             print(f"[完成] 第 {folder_index} 个文件夹处理完成，结果已保存至: {output_excel}")
#         else:
#             print(f"[提示] 第 {folder_index} 个文件夹：没有任何 gap 产生结果（可能图片数量不足）。")
#
# if __name__ == '__main__':
#     MODEL_PATH ="D:/PyCharm/RAFT-master/raft-model/ultra+C+T+S+K/raft-c+t+k+s.pth"#"D:/PyCharm/RAFT-master/raft-model/ultra+C+T/raft-chairs+things.pth"
#     BASE_IMAGE_PATH = "D://PyCharm//flownet2//OwnData//gaoya//20-25point//20-25-{}"
#     BASE_OUTPUT_EXCEL = "D://PyCharm//flownet2//OwnData//gaoya//20-25point//ultra-c+t+s+k-module-output_20-25-{}_g3-5-7.xlsx"
#
#     SMALL = False
#     MIXED_PRECISION = False
#     ALTERNATE_CORR = False
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', help="restore checkpoint")
#     parser.add_argument('--path', help="dataset for evaluation")
#     parser.add_argument('--small', action='store_true', help='use small model')
#     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#     parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
#
#     for i in range(1, 15):
#         IMAGE_PATH = BASE_IMAGE_PATH.format(i)
#         OUTPUT_EXCEL = BASE_OUTPUT_EXCEL.format(i)
#
#         arg_list = [
#             '--model', MODEL_PATH,
#             '--path', IMAGE_PATH,
#         ]
#         if SMALL:
#             arg_list.append('--small')
#         if MIXED_PRECISION:
#             arg_list.append('--mixed_precision')
#         if ALTERNATE_CORR:
#             arg_list.append('--alternate_corr')
#
#         args = parser.parse_args(arg_list)
#         demo(args, OUTPUT_EXCEL, folder_index=i)

# -*- coding: utf-8 -*-
import sys, os, glob, argparse, warnings
import numpy as np
import torch
from PIL import Image
import pandas as pd

# ---- 路径与模型 ----
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from core.raft import RAFT
from core.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== 可调参数 ======
FPS = 60.0
GSD_M_PER_PX = 0.0128      # 地面采样距离 (m/px)
GAPS = [3, 5, 7]           # 需要输出的帧间隔

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = torch.from_numpy(img).permute(2,0,1).float()
    return img[None].to(DEVICE)

def compute_flow_statistics(flow, gap, fps=FPS, gsd=GSD_M_PER_PX):
    flow_np = flow[0].detach().cpu().numpy()
    u_px, v_px = flow_np[0], flow_np[1]
    mag_px = np.sqrt(u_px**2 + v_px**2)
    scale = (fps/float(gap)) * gsd  # px/frame -> m/s
    mag_ms = mag_px * scale
    u_ms, v_ms = u_px*scale, v_px*scale
    return {
        'Mean_U': float(np.mean(u_ms)),
        'Mean_V': float(np.mean(v_ms)),
        'Mean_Magnitude': float(np.mean(mag_ms)),
        'Max_Magnitude': float(np.max(mag_ms)),
        # 也保留像素域，若不需要可删
        'Mean_U_px': float(np.mean(u_px)),
        'Mean_V_px': float(np.mean(v_px)),
        'Mean_Mag_px': float(np.mean(mag_px)),
    }

def make_pairs(images, gap):
    n = len(images)
    if n <= gap: return []
    return [(images[i], images[i+gap]) for i in range(0, n-gap)]

def build_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module.to(DEVICE).eval()
    return model

def demo(args, output_excel, folder_index, summary_rows):
    """运行一个文件夹。
    新增：把每个 gap 的两项“均值的均值”加入 summary_rows（用于全局汇总表）"""
    model = build_model(args)

    with torch.no_grad():
        exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tiff','*.gif')
        images = sorted(sum([glob.glob(os.path.join(args.path, e)) for e in exts], []))

        if len(images) < 2:
            print(f"[警告] 第 {folder_index} 个文件夹：至少需要两张图片，已跳过。")
            return

        writer = pd.ExcelWriter(output_excel, engine='openpyxl')
        any_rows = False

        for gap in GAPS:
            pairs = make_pairs(images, gap)
            if not pairs:
                print(f"[提示] 第 {folder_index} 个文件夹：gap={gap} 可用配对为空，跳过。")
                continue

            rows = []
            for imfile1, imfile2 in pairs:
                image1, image2 = load_image(imfile1), load_image(imfile2)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                stats = compute_flow_statistics(flow_up, gap=gap, fps=FPS, gsd=GSD_M_PER_PX)
                rows.append({
                    'Image1': os.path.basename(imfile1),
                    'Image2': os.path.basename(imfile2),
                    'Gap': gap,
                    **stats
                })

            if rows:
                any_rows = True
                df_gap = pd.DataFrame(rows)
                df_gap.to_excel(writer, index=False, sheet_name=f"gap_{gap}")

                mean_mean_mag = df_gap['Mean_Magnitude'].mean()
                mean_max_mag  = df_gap['Max_Magnitude'].mean()
                n_pairs = len(df_gap)

                # —— 这里的打印与你原来一致 ——
                print(f"[完成] Folder {folder_index} | gap={gap} | "
                      f"Mean(Mean_Magnitude)={mean_mean_mag:.4f} m/s | "
                      f"Mean(Max_Magnitude)={mean_max_mag:.4f} m/s")

                # —— 新增：把这两项汇总保存到 summary_rows ——
                summary_rows.append({
                    'Folder': folder_index,
                    'Gap': gap,
                    'MeanOfMean_Magnitude_ms': float(mean_mean_mag),
                    'MeanOfMax_Magnitude_ms': float(mean_max_mag),
                    'N_pairs': int(n_pairs),
                })

        writer.close()
        if any_rows:
            print(f"[完成] 第 {folder_index} 个文件夹处理完成，结果已保存至: {output_excel}")
        else:
            print(f"[提示] 第 {folder_index} 个文件夹：没有任何 gap 产生结果。")

if __name__ == '__main__':
    # ===== 路径配置（按需修改） =====
    MODEL_PATH = "D:/PyCharm/RAFT-master/raft-model/ultra+C+T/raft-chairs+things.pth"#"D:/PyCharm/RAFT-master/raft-model/ultrta+C+T+SorK/100000_raft-c+t-kitti.pth"#"D:/PyCharm/RAFT-master/raft-model/ultra+C+T+S+K/raft-c+t+k+s.pth"
    BASE_IMAGE_PATH = "D://PyCharm//flownet2//OwnData//zhangye//back\zhangye301//20-25point//20-25-{}"
    BASE_OUTPUT_EXCEL = "D://PyCharm//flownet2//OwnData//zhangye//back\zhangye301//20-25point//ultra-c+t+k-module-output_20-25-{}_g3-5-7.xlsx"

    # —— 新增：一个“全局汇总”的导出文件 ——
    GLOBAL_SUMMARY_XLSX = "D://PyCharm//flownet2//OwnData//zhangye//back\zhangye301//20-25point////SUMMARY_gap-c+t_3_5_7.xlsx"

    # RAFT flags
    SMALL = False; MIXED_PRECISION = False; ALTERNATE_CORR = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')

    summary_rows = []   # <<<<<< 新增：收集打印统计用于汇总表

    # 逐文件夹运行
    for i in range(1, 15):
        IMAGE_PATH = BASE_IMAGE_PATH.format(i)
        OUTPUT_EXCEL = BASE_OUTPUT_EXCEL.format(i)

        arg_list = ['--model', MODEL_PATH, '--path', IMAGE_PATH]
        if SMALL: arg_list.append('--small')
        if MIXED_PRECISION: arg_list.append('--mixed_precision')
        if ALTERNATE_CORR: arg_list.append('--alternate_corr')
        args = parser.parse_args(arg_list)

        demo(args, OUTPUT_EXCEL, folder_index=i, summary_rows=summary_rows)

    # ====== 汇总成一个表格 ======
    if len(summary_rows) == 0:
        print("[结束] 没有任何汇总数据。")
        sys.exit(0)

    df_sum = pd.DataFrame(summary_rows)

    # 宽表：行=Folder，列=各 gap 的两项统计
    wide_meanmag = df_sum.pivot(index='Folder', columns='Gap', values='MeanOfMean_Magnitude_ms')
    wide_maxmag  = df_sum.pivot(index='Folder', columns='Gap', values='MeanOfMax_Magnitude_ms')
    # 为了可读，给列加层级标题
    wide_meanmag.columns = [f"MeanMag(k={c}) [m/s]" for c in wide_meanmag.columns]
    wide_maxmag.columns  = [f"MaxMag(k={c}) [m/s]"  for c in wide_maxmag.columns]
    wide_pivot = pd.concat([wide_meanmag, wide_maxmag], axis=1).sort_index(axis=1)

    # 导出到一个 Excel（两个 sheet）
    with pd.ExcelWriter(GLOBAL_SUMMARY_XLSX, engine='openpyxl') as writer:
        df_sum.to_excel(writer, index=False, sheet_name='by_folder_gap')   # 长表
        wide_pivot.to_excel(writer, sheet_name='wide_pivot')              # 宽表（更像论文表）

    print(f"[完成] 已将控制台打印的统计汇总到：{GLOBAL_SUMMARY_XLSX}")
