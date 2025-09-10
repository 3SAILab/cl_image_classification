import matplotlib.pyplot as plt
import numpy as np
import os
import json
from cycler import cycler

def plot_loss_curves(loss_list, epochs, save_path):
    plt.figure(figsize=(10, 7))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(range(epochs), loss_list, label="train_loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_curves(accuracy_list, epochs, save_path):
    plt.figure(figsize=(10, 7))
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.plot(range(epochs), accuracy_list, label="accuracy")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(loss, acc, val_loss, val_acc, title, name, figsize=(12, 6)):
    # --- 数据准备 ---
    epochs = range(1, len(loss) + 1)

    loss = np.array(loss)
    val_loss = np.array(val_loss)

    loss = (loss - loss.min()) / (loss.max() - loss.min())
    val_loss = (val_loss - val_loss.min()) / (val_loss.max() - val_loss.min())

    # 找到最佳验证准确率点
    best_epoch = np.argmax(val_acc) + 1
    best_val_acc = np.max(val_acc)
    
    # --- 绘图设置 ---
    # 使用更适合论文的'seaborn-v0_8-paper'风格，并设置全局字体
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 12 # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 12 # y轴刻度字体大小
    plt.rcParams['legend.fontsize'] = 12 # 图例字体大小
    plt.rcParams['figure.titlesize'] = 16 # 总标题字体大小

    fig, ax1 = plt.subplots(figsize=figsize)
    
    # --- 绘制损失曲线 (主Y轴) ---
    # 使用柔和且对比鲜明的颜色
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss, fontsize=14)
    ln1 = ax1.plot(epochs, loss, color=color_loss, linestyle='-', label='Training Loss')
    ln2 = ax1.plot(epochs, val_loss, color=color_loss, linestyle='--', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(linestyle=':', linewidth=0.5) # 添加虚线网格

    # --- 绘制准确率曲线 (次Y轴) ---
    ax2 = ax1.twinx()  # 共享X轴，创建次Y轴
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color_acc, fontsize=14)
    ln3 = ax2.plot(epochs, acc, color=color_acc, linestyle='-', label='Training Accuracy')
    ln4 = ax2.plot(epochs, val_acc, color=color_acc, linestyle='--', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # --- 添加最佳点标注 ---
    ax2.annotate(f'Best Val Acc: {best_val_acc:.4f}\nEpoch: {best_epoch}',
                 xy=(best_epoch, best_val_acc),
                 xytext=(best_epoch + 3, best_val_acc + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.5))
    # 在最佳点上画一个标记
    ax2.plot(best_epoch, best_val_acc, 'o', color='gold', markersize=8, markeredgecolor='black', label=f'Best Epoch ({best_epoch})')

    # --- 图例和标题 ---
    # 合并两个Y轴的图例
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    # --- [修改] 调整图例位置到右侧中间 ---
    ax1.legend(lns, labs, loc='right', frameon=True, shadow=True, borderpad=1)
    
    plt.title(title, pad=20) # 设置标题并增加间距

    # 调整布局，防止标签重叠
    fig.tight_layout()

    plt.savefig("results/{}_curves.png".format(name))
    
    plt.show()

def plot_multiple_model_comparison(model_data, metric_to_plot='accuracy', title=None, figsize=(12, 8)):
    """
    在同一图表中对比多个模型在多次运行下的平均性能，并显示置信区间。

    参数:
    model_data (dict): 字典，键为模型名称(str)，值为该模型的histories列表(list of dict)。
                       例如: {'ResNet50': [hist1, hist2], 'VGG16': [hist3, hist4]}
    metric_to_plot (str): 要绘制的指标，可选值为 'accuracy' 或 'loss'。
    title (str): 图表的标题。如果为None，则会自动生成。
    figsize (tuple): 图表的大小。
    """
    
    # --- 绘图设置 ---
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16
    
    # 设置更丰富的颜色循环
    color_cycle = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.rc('axes', prop_cycle=color_cycle)

    fig, ax = plt.subplots(figsize=figsize)
    
    # --- 确定要绘制的指标 ---
    if metric_to_plot == 'accuracy':
        train_metric_key = 'Accuracy List'
        val_metric_key = 'Val Accuracy List'
        y_label = 'Accuracy'
    elif metric_to_plot == 'loss':
        train_metric_key = 'Loss List'
        val_metric_key = 'Val Loss List'
        y_label = 'Loss'
    else:
        print("Error: metric_to_plot must be 'accuracy' or 'loss'.")
        return

    # --- 遍历每个模型的数据并绘图 ---
    for model_name, histories in model_data.items():
        if not histories:
            print(f"Warning: No history data for model '{model_name}'. Skipping.")
            continue
            
        # 聚合数据并计算均值/标准差
        all_val_metric = np.array([h[val_metric_key] for h in histories])
        mean_val_metric = np.mean(all_val_metric, axis=0)
        std_val_metric = np.std(all_val_metric, axis=0)
        
        epochs = range(1, len(mean_val_metric) + 1)
        
        # 绘制验证集的均值曲线
        line, = ax.plot(epochs, mean_val_metric, linestyle='-', label=f'{model_name} (Validation)')
        # 填充验证集的置信区间
        ax.fill_between(epochs, mean_val_metric - std_val_metric, mean_val_metric + std_val_metric, alpha=0.2, color=line.get_color())

    # --- 图表美化 ---
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.grid(linestyle=':', linewidth=0.5)
    
    # 自动生成标题
    if title is None:
        title = f'Comparison of Model Validation {y_label}'
    
    plt.title(title, pad=20)
    
    # --- [修改] 将图例移动到图表下方，并横向排列 ---
    num_models = len(model_data)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), # 将图例的锚点放在图表下方
        ncol=num_models,             # 根据模型数量设置列数，使其横向排列
        fancybox=True, 
        shadow=True
    )
    
    # 调整布局以适应图例
    fig.tight_layout()
    
    # --- 保存和显示 ---
    # savefig中的 bbox_inches='tight' 会自动调整边界框以包含图例
    plt.savefig("results/model_comparison_1.png")
    plt.show()

def make_models_data(models, num):
    models_data = {}
    for k in range(len(models)):
        histories = []
        for i in range(1, num+1):
            log_name = "{}_{}.txt".format(models[k], i)
            log_path = os.path.join("logs", log_name)
            assert os.path.exists(log_path),"file {} does not exist.".format(log_path)

            with open(log_path, "r") as f:
                history = json.load(f)
            histories.append(history)
        
        models_data[models[k]] = histories
    
    return models_data
        

if __name__ == "__main__":
    
    num_log = 3 # 取几份训练结果文件
    models = ['GoogLeNet', 'InceptionResNetV2'] # 对比哪些模型
    models_data = make_models_data(models, num_log)

    plot_multiple_model_comparison(models_data, metric_to_plot='loss')


    # name = "densenet121"
    # log_path = "logs/{}.txt".format(name)
    # assert os.path.exists(log_path),"file {} does not exist.".format(log_path)
    # with open(log_path, "r") as f:
    #     history = json.load(f)

    # plot_training_curves(history['Loss List'], history['Accuracy List'], history['Val Loss List'], history['Val Accuracy List'],
    #                      title="Image Classification Model Performance", name=name)