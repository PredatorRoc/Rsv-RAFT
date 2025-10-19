import os
import pandas as pd


def process_xlsx_files(directory, file_ranges):
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        # 仅处理xlsx文件
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)
            # 获取当前文件的范围
            if filename in file_ranges:
                min_value, max_value = file_ranges[filename]
            else:
                # 如果没有指定范围，默认范围为0到100
                min_value, max_value = 0, 100

            # 读取xlsx文件
            df = pd.read_excel(file_path)

            print(f"Columns in {filename}: {df.columns.tolist()}")

            # 检查C和D列是否存在
            if 'Mean_Magnitude' in df.columns and 'Max_Magnitude' in df.columns:
                # 计算C列和D列的平均值
                avg_c = df['Mean_Magnitude'].mean()
                avg_d = df['Max_Magnitude'].mean()

                # 计算C列和D列在指定范围内的个数
                count_c = ((df['Mean_Magnitude'] >= min_value) & (df['Mean_Magnitude'] <= max_value)).sum()
                count_d = ((df['Max_Magnitude'] >= min_value) & (df['Max_Magnitude'] <= max_value)).sum()

                # 输出结果
                print(f"File: {filename}")
                print(f"Average of column Mean_Magnitude: {avg_c}")
                print(f"Average of column Max_Magnitude: {avg_d}")
                print(f"Count of values in range [{min_value}, {max_value}] in column Mean_Magnitude: {count_c}")
                print(f"Count of values in range [{min_value}, {max_value}] in column Max_Magnitude: {count_d}")
                print("-" * 40)
            else:
                print(f"File: {filename} does not contain columns Mean_Magnitude and Max_Magnitude.")


# 设置文件夹路径以及每个文件对应的范围
directory = "D:/PyCharm/flownet2/OwnData/liancheng1126/20-25point"  # 替换为你文件夹的路径

# 文件名对应的范围字典，键为文件名，值为一个元组(min_value, max_value)
file_ranges = {
    "output_20-25-1.xlsx": (0.5899, 0.7981),
    "output_20-25-2.xlsx": (0.4896, 0.6624),
    "output_20-25-3.xlsx": (0.41565, 0.56235),
    "output_20-25-4.xlsx": (1.4824, 2.0056),
    "output_20-25-5.xlsx": (0.8075, 1.0925),
    "output_20-25-6.xlsx": (0.57885,0.78315),
    "output_20-25-7.xlsx": (0.5474, 0.7406),
    "output_20-25-8.xlsx": (0.7735, 1.0465),
    "output_20-25-9.xlsx": (0.76245, 1.03155),
    "output_20-25-10.xlsx": (0.5797, 0.7843),
    "output_20-25-11.xlsx": (0.4743, 0.6417),
    "output_20-25-12.xlsx": (0.5559, 0.7521),
    "output_20-25-13.xlsx": (0.34085, 0.46115),
    "output_20-25-14.xlsx": (0.272, 0.368),
    # 你可以继续添加其他文件及其对应范围
}

# 调用函数处理文件
process_xlsx_files(directory, file_ranges)
