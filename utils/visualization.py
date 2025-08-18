import matplotlib.pyplot as plt
import numpy as np
import os
import json

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

def plot_training_curves(loss, acc, val_loss, val_acc, title, figsize=(12, 6)):
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
    
    plt.show()

if __name__ == "__main__":
    name = "GoogLeNet"
    log_path = "logs/{}.txt".format(name)
    assert os.path.exists(log_path),"file {} does not exist.".format(log_path)
    with open(log_path, "r") as f:
        history = json.load(f)

    plot_training_curves(history['Loss List'], history['Accuracy List'], history['Val Loss List'], history['Val Accuracy List'],
                         title="Image Classification Model Performance")

    plt.savefig("results/{}_curves.png".format(name))
