import matplotlib.pyplot as plt

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