import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import jos3
from jos3.utilities import local_clo_typical_ensembles

directory_name = "jos3_output_example"
current_directory = os.getcwd()
jos3_example_directory = os.path.join(current_directory, directory_name)
if not os.path.exists(jos3_example_directory):
    os.makedirs(jos3_example_directory)

# 创建模型实例
model = jos3.JOS3(
    height=1.736,
    weight=69.1,
    fat=22.9,
    age=24.5,
    sex="male",
    bmr_equation="ganpule",
    bsa_equation="fujimoto",
    ex_output="all",
)

# 设置环境条件 - 支持多段环境模拟
print("设置环境条件（分段模拟）...")

# 每个字典代表一段环境条件，持续时间单位为分钟
environment_segments = [
    {
        "name": "初始温和环境",
        "duration_minutes": 35,
        "Ta": 24.5,
        "Tr": 24.5,
        "RH": 65,
        "Va": 0.2,
    },
    {
        "name": "初始温和环境",
        "duration_minutes": 35,
        "Ta": 24.5,
        "Tr": 24.5,
        "RH": 65,
        "Va": 0.2,
    },

]

# 模拟的时间步长（秒）
dtime = 60

# 设置模型输入参数
model.load_mass = 50  # 负载质量 50kg
model.march_speed = 0  # 行军速度 1.11 m/s
model.snow_depth = 0  # 雪厚度 1.0m
model.slope = 5  # 坡度 5%

# 设置服装热阻
try:
    clothing_entry = local_clo_typical_ensembles["xianyi1"]

    body_parts_order = [
        "head", "neck", "chest", "back", "pelvis",
        "left_shoulder", "left_arm", "left_hand",
        "right_shoulder", "right_arm", "right_hand",
        "left_thigh", "left_leg", "left_foot",
        "right_thigh", "right_leg", "right_foot"
    ]

    local_body = clothing_entry.get("local_body_part", {})
    clo_values = []
    ret_values = []
    have_local_clo = True
    have_local_ret = True

    if isinstance(local_body, dict):
        for part in body_parts_order:
            part_info = local_body.get(part)
            if isinstance(part_info, dict):
                if "clo" in part_info:
                    clo_values.append(part_info["clo"])
                else:
                    have_local_clo = False
                    break
                if "ret" in part_info:
                    ret_values.append(part_info["ret"])
                else:
                    have_local_ret = False
            elif isinstance(part_info, (int, float)):
                clo_values.append(part_info)
                have_local_ret = False
            else:
                have_local_clo = False
                break

    if have_local_clo and len(clo_values) == len(body_parts_order):
        model.Icl = clo_values
        if have_local_ret and len(ret_values) == len(body_parts_order):
            model.Iret = ret_values
            ret_message = "并使用提供的局部湿阻"
        else:
            model.Iret = None
            ret_message = "，湿阻将根据热阻自动推算"
        whole_clo = clothing_entry.get("whole_body")
        if whole_clo is not None:
            print(f"服装热阻设置成功，局部数据来自字典，参考全身热阻 {whole_clo} clo{ret_message}")
        else:
            print(f"服装热阻设置成功，局部数据来自字典{ret_message}")
    elif "whole_body" in clothing_entry:
        model.Icl = clothing_entry["whole_body"]
        whole_ret = clothing_entry.get("whole_body_ret")
        if whole_ret is not None:
            model.Iret = whole_ret
            ret_message = "并使用 whole_body_ret 指定的湿阻"
        else:
            model.Iret = None
            ret_message = "，湿阻将根据热阻自动推算"
        print(f"仅提供全身热阻，已按体表面积自动分配至各部位{ret_message}")
    else:
        raise KeyError("缺少局部或全身服装热阻数据")

except KeyError as e:
    print(f"错误：找不到指定的服装组合或数据缺失 - {e}")
    print("可用的服装组合：")
    for key in local_clo_typical_ensembles.keys():
        print(f"  - {key}")
# 设置活动水平和姿势
model.posture = "sitting"

print("开始模拟...")

segment_summary = []
total_minutes = 0.0

for idx, segment in enumerate(environment_segments, start=1):
    name = segment.get("name", f"环境段 {idx}")
    duration_minutes = float(segment.get("duration_minutes", 0))

    if duration_minutes <= 0:
        print(f"- 跳过{name}：持续时间为0")
        continue

    steps = int(np.ceil(duration_minutes * 60 / dtime))
    simulated_minutes = steps * dtime / 60

    print(
        f"- 第 {idx} 段 {name}："
        f"目标 {duration_minutes:.1f} 分钟，实际模拟 {simulated_minutes:.1f} 分钟"
    )

    if "Ta" in segment:
        model.Ta = segment["Ta"]
    if "Tr" in segment:
        model.Tr = segment["Tr"]
    elif "Ta" in segment:
        model.Tr = segment["Ta"]
    if "RH" in segment:
        model.RH = segment["RH"]
    if "Va" in segment:
        model.Va = segment["Va"]

    model.simulate(times=steps, dtime=dtime)

    segment_summary.append(
        {
            "name": name,
            "steps": steps,
            "configured_minutes": duration_minutes,
            "simulated_minutes": simulated_minutes,
        }
    )
    total_minutes += simulated_minutes

if segment_summary:
    print(f"所有环境段模拟完成，总模拟时长约为 {total_minutes:.1f} 分钟。")
else:
    print("未配置有效的环境段，仅保留模型初始稳态。")

# 获取结果并验证数据
print("获取模拟结果...")
df = pd.DataFrame(model.dict_results())

# 添加时间与环境段信息，便于后续分析
df["ElapsedMinutes"] = df.index * (dtime / 60)

segment_labels = ["初始条件"]
for info in segment_summary:
    segment_labels.extend([info["name"]] * info["steps"])

if len(segment_labels) < len(df):
    segment_labels.extend([segment_labels[-1]] * (len(df) - len(segment_labels)))

df["EnvironmentSegment"] = segment_labels[:len(df)]

print("环境段模拟总结:")
if segment_summary:
    cumulative_minutes = 0.0
    for info in segment_summary:
        cumulative_minutes += info["simulated_minutes"]
        print(
            f"  - {info['name']}: 设定 {info['configured_minutes']:.1f} 分钟，"
            f"实际模拟 {info['simulated_minutes']:.1f} 分钟，累计 {cumulative_minutes:.1f} 分钟"
        )
else:
    print("  - 仅包含初始稳态数据")

# 详细的数据检查
print("=" * 50)
print("数据检查结果:")
print("=" * 50)
print(f"数据框形状: {df.shape}")
print(f"数据框列名: {list(df.columns)}")
print(f"数据框前3行:")
print(df.head(3))

# 检查关键温度列是否存在
required_columns = ['TskMean', 'TcrHead', 'TskHead', 'TskChest', 'TskBack', 'TskLHand', 'TskLFoot']
available_columns = df.columns.tolist()

print(f"\n需要的列: {required_columns}")
print(f"可用的列: {[col for col in available_columns if any(req in col for req in ['Tsk', 'Tcr'])]}")

# 检查数据是否有效
print(f"\n数据有效性检查:")
print(f"TskMean 非空值数量: {df['TskMean'].notna().sum()}")
print(f"TskMean 数值范围: [{df['TskMean'].min():.2f}, {df['TskMean'].max():.2f}]")

# 检查是否有温度数据
if 'TskMean' in df.columns and df['TskMean'].notna().any():
    print("温度数据有效，开始绘图...")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 第一张图：平均皮肤温度和核心温度
    if 'TskMean' in df.columns and 'TcrHead' in df.columns:
        ax1.plot(df['ElapsedMinutes'], df['TskMean'], label='平均皮肤温度', linewidth=2)
        ax1.plot(df['ElapsedMinutes'], df['TcrHead'], label='头部核心温度', linewidth=2)
        ax1.set_ylabel("温度 [°C]")
        ax1.set_xlabel("模拟时间 [分钟]")
        ax1.set_title("皮肤和核心温度变化")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        print("缺少 TskMean 或 TcrHead 列")

    # 第二张图：局部皮肤温度
    skin_columns = ['TskHead', 'TskChest', 'TskBack', 'TskLHand', 'TskLFoot']
    available_skin_cols = [col for col in skin_columns if col in df.columns]

    if available_skin_cols:
        for col in available_skin_cols:
            ax2.plot(df['ElapsedMinutes'], df[col], label=col.replace('Tsk', ''))
        ax2.set_ylabel("皮肤温度 [°C]")
        ax2.set_xlabel("模拟时间 [分钟]")
        ax2.set_title("局部皮肤温度变化")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        print("缺少皮肤温度数据列")
        # 尝试绘制任何可用的温度数据
        temp_cols = [col for col in df.columns if 'Tsk' in col or 'Tcr' in col]
        if temp_cols:
            for col in temp_cols[:5]:  # 只绘制前5个温度列
                ax2.plot(df['ElapsedMinutes'], df[col], label=col)
            ax2.set_ylabel("温度 [°C]")
            ax2.set_xlabel("模拟时间 [分钟]")
            ax2.set_title("可用温度数据")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    if segment_summary:
        boundary_minutes = np.cumsum([info["steps"] for info in segment_summary]) * (dtime / 60)
        for boundary in boundary_minutes[:-1]:
            for axis in (ax1, ax2):
                axis.axvline(boundary, color='gray', linestyle='--', alpha=0.4)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(jos3_example_directory, "jos3_temperatures.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.show()

else:
    print("错误：没有有效的温度数据可供绘图")
    print("可用的数值列:", [col for col in df.columns if df[col].dtype in ['float64', 'int64']])

# 导出结果为CSV
csv_path = os.path.join(jos3_example_directory, "jos3_results.csv")
model.to_csv(csv_path)
print(f"CSV结果已保存到: {csv_path}")

# 额外调试：显示所有可用的数据列
print("\n所有可用数据列:")
for i, col in enumerate(df.columns):
    print(f"{i + 1:2d}. {col}")