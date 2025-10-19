# import os
# import cv2
# import numpy as np
#
# def horn_schunck(I1, I2, alpha=1.0, N_iter=100):
#     I1 = I1.astype(np.float32) / 255.0
#     I2 = I2.astype(np.float32) / 255.0
#
#     Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3)
#     Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3)
#     It = I2 - I1
#
#     u = np.zeros_like(I1)
#     v = np.zeros_like(I1)
#
#     kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
#                        [1 / 6, 0, 1 / 6],
#                        [1 / 12, 1 / 6, 1 / 12]], dtype=np.float32)
#
#     for _ in range(N_iter):
#         u_avg = cv2.filter2D(u, -1, kernel)
#         v_avg = cv2.filter2D(v, -1, kernel)
#
#         der = (Ix * u_avg + Iy * v_avg + It) / (alpha ** 2 + Ix ** 2 + Iy ** 2)
#         u = u_avg - Ix * der
#         v = v_avg - Iy * der
#
#     return u, v
#
# def process_image_folder(image_folder, alpha=1.0, N_iter=100):
#     pixel_to_meter = 0.0128
#     fps = 30.0
#     scale = pixel_to_meter * fps
#
#     image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)
#                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
#
#     if len(image_files) < 2:
#         print(f" 文件夹 {os.path.basename(image_folder)} 中图片不足两张，跳过。")
#         return None
#
#     total_speeds = []
#     max_speeds = []
#
#     print(f"\n 开始处理文件夹：{os.path.basename(image_folder)}")
#
#     for i in range(len(image_files) - 1):
#         img1 = cv2.imread(image_files[i])
#         img2 = cv2.imread(image_files[i + 1])
#         gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
#         u, v = horn_schunck(gray1, gray2, alpha=alpha, N_iter=N_iter)
#
#         mag = np.sqrt(u ** 2 + v ** 2)
#         mag_mps = mag * scale
#
#         avg_speed = np.mean(mag_mps)
#         max_speed = np.max(mag_mps)
#
#         total_speeds.append(avg_speed)
#         max_speeds.append(max_speed)
#
#     overall_avg = np.mean(total_speeds)
#     overall_max = np.max(max_speeds)
#
#     print(f" 平均速度: {overall_avg:.4f} m/s | 最大速度: {overall_max:.4f} m/s")
#
#     return os.path.basename(image_folder), overall_avg, overall_max
#
# def process_all_folders(parent_folder, alpha=1.0, N_iter=100):
#     print(f" 扫描父目录：{parent_folder}")
#     results = []
#
#     for subfolder in sorted(os.listdir(parent_folder)):
#         subfolder_path = os.path.join(parent_folder, subfolder)
#         if os.path.isdir(subfolder_path) and subfolder.startswith("20-25-"):
#             result = process_image_folder(subfolder_path, alpha, N_iter)
#             if result:
#                 results.append(result)
#
#     # 汇总打印
#     print("\n 所有文件夹速度统计结果：")
#     for name, avg, maxv in results:
#         print(f"   {name}: 平均速度 = {avg:.4f} m/s, 最大速度 = {maxv:.4f} m/s")
#
# if __name__ == "__main__":
#     parent_folder_path = r"D:/PyCharm/flownet2/OwnData/gaoya/20-25point"
#
#     alpha = 1.0
#     N_iter = 10
#
#     process_all_folders(parent_folder_path, alpha, N_iter)
#
#
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# def generate_hsv_plane(width=512, height=512):
#     """
#     生成 HSV 空间的 H-S 平面（V=1）图像。
#     Hue 从左到右（0-360°），Saturation 从上到下（0-1）
#     """
#     H = np.linspace(0, 180, width, dtype=np.float32)  # OpenCV Hue: 0-180
#     S = np.linspace(0, 1, height, dtype=np.float32)
#
#     hsv = np.zeros((height, width, 3), dtype=np.float32)
#     for i in range(height):
#         hsv[i, :, 0] = H              # Hue 横向变化
#         hsv[i, :, 1] = S[i]           # Saturation 纵向变化
#         hsv[i, :, 2] = 1.0            # Value 恒定为1
#
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     rgb = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
#
#     return rgb
#
# # 显示图像
# hsv_plane = generate_hsv_plane(512, 512)
# plt.imshow(hsv_plane)
# plt.axis('off')
# plt.title("HSV H-S Plane (V=1)")
# plt.show()
#

import os
import cv2
import numpy as np
from glob import glob

def horn_schunck(I1, I2, alpha=1.0, N_iter=100):
    I1 = I1.astype(np.float32) / 255.0
    I2 = I2.astype(np.float32) / 255.0

    Ix = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3)
    It = I2 - I1

    u = np.zeros_like(I1)
    v = np.zeros_like(I1)

    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,  0,   1/6 ],
                       [1/12, 1/6, 1/12]], dtype=np.float32)

    for _ in range(N_iter):
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)
        der = (Ix * u_avg + Iy * v_avg + It) / (alpha**2 + Ix**2 + Iy**2 + 1e-5)
        u = u_avg - Ix * der
        v = v_avg - Iy * der

    return u, v


def flow_to_color(u, v, threshold=0.1):
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)

    hsv[..., 0] = (angle / 2).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 将光流幅值小于 threshold 的像素设为白色
    mask = magnitude < threshold
    bgr[mask] = [255, 255, 255]

    return bgr


def process_image_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(glob(os.path.join(input_folder, '*.*')))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) < 2:
        print("⚠ 图片数量不足两张，无法计算光流。")
        return

    for i in range(len(image_files) - 1):
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + 1])

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        u, v = horn_schunck(gray1, gray2)
        flow_color = flow_to_color(u, v)

        out_name = f"flow_{i:04d}.png"
        cv2.imwrite(os.path.join(output_folder, out_name), flow_color)
        print(f" 已保存光流图：{out_name}")

# 示例调用（可修改为你的路径）
if __name__ == '__main__':
    input_dir = r'D:\PyCharm\flownet2\OwnData\zhangye\back\zhangye301\10-15point\10-15-7'           # 输入文件夹路径
    output_dir = r'D:\PyCharm\flownet2\OwnData\zhangye\back\zhangye301\10-15point\10-15HS'    # 输出文件夹路径
    process_image_folder(input_dir, output_dir)

