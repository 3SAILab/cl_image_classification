import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools
import os
import csv

class ConfusionMatrix(object):
    def __init__(self, num_classes, labels, normalize, batch_size, log_name):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.normalize = normalize
        self.batch_size = batch_size
        self.model_name = log_name[:-2]
        self.save_path_matrix = "results/{}_matrix.png".format(log_name)
        self.save_path_csv = "results/model_classifier_report.csv"

    # 更新数据
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        total_samples = np.sum(self.matrix)
        acc = sum_TP / total_samples if total_samples != 0 else 0.0
 
        table = PrettyTable()
        table.field_names = ["模型名", "类别", "Precision", "Recall", "Specificity", "全局准确率"]
        csv_rows = []

        report_content = []
        report_content.append(f"模型准确率: {acc:.4f}\n")
        report_content.append(str(table.field_names) + "\n")

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            csv_rows.append([
                self.log_name,       # 模型名
                self.labels[i],      # 类别名
                Precision,           # 精确率
                Recall,              # 召回率
                Specificity,         # 特异性
                round(acc, 4),       # 全局准确率
                total_samples        # 总样本数
            ])
            

        self._write_to_csv(csv_rows)

    # 分类报告写入csv文件
    def _write_to_csv(self, csv_rows):
        csv_dir = os.path.dirname(self.csv_path)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        csv_headers = [
            "模型名", "类别", "Precision(精确率)", "Recall(召回率)", 
            "Specificity(特异性)", "全局准确率", "总验证样本数"
        ]

        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, mode='a', encoding='utf-8', newline='') as f:  # mode='a' 表示追加模式
            writer = csv.writer(f)
            # 首次写入时添加表头
            if not file_exists:
                writer.writerow(csv_headers)
            # 追加当前模型的所有类别数据
            writer.writerows(csv_rows)
            writer.writerow([""] * len(csv_headers))
        

    def plot(self):
        self.plot_confusion_matrix()
 
    def plot_confusion_matrix(self):
        """
         - matrix : 计算出的混淆矩阵的值
         - classes : 混淆矩阵中每一行每一列对应的列
         - normalize : True:显示百分比, False:显示个数
        """
        matrix = self.matrix
        classes = self.labels
        normalize = self.normalize
        title = 'Confusion matrix'
        cmap = plt.cm.Blues  # 绘制的颜色

        fig_size = (min(20, 5 + len(classes)*0.8), min(18, 4 + len(classes)*0.8))
        plt.figure(figsize=fig_size)

        if normalize:
            row_sums = matrix.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1
            matrix = matrix.astype('float') / row_sums
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else '.0f'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # 保存混淆矩阵图像
        plt.savefig(self.save_path_matrix, dpi=300, bbox_inches='tight')

        plt.close()
