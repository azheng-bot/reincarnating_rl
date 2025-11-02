import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

experiment_path = "C:\\Users\\76361\\Desktop\\my_reincarnating_rl\\result\\breakout_record\\experiment_record\\breakout\\qdagger_rainbow"



def load_tensorboard_metrics(experiment_path):
    """从metrics/tensorboard加载TensorBoard日志"""
    tb_path = os.path.join(experiment_path, "metrics", "tensorboard")
    if not os.path.exists(tb_path):
        raise FileNotFoundError(f"TensorBoard目录不存在: {tb_path}")
    
    # 查找最新的TensorBoard事件文件
    event_files = sorted(
        [f for f in os.listdir(tb_path) if f.startswith("events.out.tfevents")],
        reverse=True
    )
    
    if not event_files:
        raise FileNotFoundError(f"未找到TensorBoard事件文件: {tb_path}")
    
    event_file = os.path.join(tb_path, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    return ea

# 使用示例
ea = load_tensorboard_metrics(experiment_path)

# 获取所有可用的指标标签
print("可用指标:", ea)





def load_console_logs(experiment_path):
    """从metrics/console加载控制台日志"""
    console_path = os.path.join(experiment_path, "metrics", "console")
    if not os.path.exists(console_path):
        raise FileNotFoundError(f"Console目录不存在: {console_path}")
    
    log_files = glob.glob(os.path.join(console_path, "*.log"))
    if not log_files:
        raise FileNotFoundError(f"未找到日志文件: {console_path}")
    
    logs = []
    for file in log_files:
        with open(file, 'r') as f:
            logs.append(f.read())
    
    return logs

# 使用示例
console_logs = load_console_logs(experiment_path)
print("加载的日志文件数量:", len(console_logs))





def check_available_tags(experiment_path):
    """检查TensorBoard日志中可用的所有指标标签"""
    tb_path = os.path.join(experiment_path, "metrics", "tensorboard")
    
    # 查找TensorBoard事件文件
    event_files = [f for f in os.listdir(tb_path) if f.startswith("events.out.tfevents")]
    if not event_files:
        print("未找到TensorBoard事件文件")
        return []
    
    event_file = os.path.join(tb_path, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    print("可用的标量指标标签:")
    for tag in ea.Tags()['scalars']:
        print(f" - {tag}")
    
    return ea.Tags()['scalars']

# 检查您的实验数据中有哪些标签
available_tags = check_available_tags(experiment_path)
print("标签总数:", len(available_tags), "标签列表:", available_tags)



def inspect_tensorboard_data(experiment_path):
    """详细检查TensorBoard数据的完整性"""
    tb_path = os.path.join(experiment_path, "metrics", "tensorboard")
    event_files = [f for f in os.listdir(tb_path) if f.startswith("events.out.tfevents")]
    
    if not event_files:
        print("错误：未找到任何TensorBoard事件文件")
        print(f"检查路径: {tb_path}")
        print(f"该目录内容: {os.listdir(tb_path)}")
        return
    
    event_file = os.path.join(tb_path, event_files[0])
    print(f"分析文件: {event_file}")
    print(f"文件大小: {os.path.getsize(event_file)} 字节")
    
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    print("\n详细的标签信息:")
    for tag_type, tags in ea.Tags().items():
        print(f"{tag_type}:", "标签列表:", tags)
        # for tag in tags:
        #     print(f"  - {tag}")
    
    # 检查特定标签的详细内容
    print("\n尝试查找包含'Average'或'Return'的标签:")
    for tag in ea.Tags()['tensors']:
        # 判断tag是不是列表
        if 'average' in tag.lower() or 'return' in tag.lower():
            data = ea.Tensors(tag)
            print(f"标签 '{tag}': 有 {len(data)} 个数据点",data[0])
            if len(data) > 0:
                print(f"  第一个值: 步数={data[0].step}, 值={data[0].tensor_content}")
                print(f"  最后一个值: 步数={data[-1].step}, 值={data[-1].tensor_content}")

# 运行详细检查
inspect_tensorboard_data(experiment_path)




# 从TensorBoard数据创建图表
def plot_tensorboard_metrics(ea):
    plt.figure(figsize=(12, 6))
    
    # 获取训练回报数据
    train_data = ea.Scalars('Train/AverageReturns')
    train_steps = [d.step for d in train_data]
    train_values = [d.value for d in train_data]
    
    # 获取评估回报数据
    eval_data = ea.Scalars('Eval/AverageReturns')
    eval_steps = [d.step for d in eval_data]
    eval_values = [d.value for d in eval_data]
    
    # 绘制图表
    plt.plot(train_steps, train_values, label='训练回报')
    plt.plot(eval_steps, eval_values, label='评估回报', linestyle='--')
    
    plt.xlabel('训练步数')
    plt.ylabel('平均回报')
    plt.title('训练与评估回报对比')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用示例
plot_tensorboard_metrics(ea)

