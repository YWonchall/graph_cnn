import matplotlib.pyplot as plt

def plot_loss_curve(loss_values, output_path):
    plt.figure()
    plt.plot([e for e in range(len(loss_values))], [float(t) for t in loss_values], label="train_loss", c='blue')
    # 添加标题和标签
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # 添加图例
    plt.legend()
    # 显示网格
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)

def plot_metric_curve(metric_values, output_path, test_interval=1):
    plt.figure()
    for name, values in metric_values.items():
        plt.plot([(e + 1) * test_interval for e in range(len(values))], [float(t) for t in values], label=name)
    # 添加标题和标签
    plt.title("Metric")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    # 添加图例
    plt.legend()
    # 显示网格
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)

def plot_result_distribution(gt, pred, output_path, top_k=100):
    gt, pred = gt[:top_k], pred[:top_k]
    plt.figure()
    plt.scatter(gt, pred, alpha=0.4)
    plt.plot(gt, gt, alpha=0.4, color='black')
    plt.xlabel("Label")
    plt.ylabel("Pred")
    # 显示网格
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)

# plt.rcParams["figure.figsize"] = (10, 6)
# plt.suptitle(args.exp_name, fontsize=16)

# plt.subplot(1, 2, 1)
# plt.ylim([0, 10])
# plt.plot([e for e in range(len(train_loss))], [float(t) for t in train_loss], label="train_loss", c='blue')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# mae_test = 'MAE : ' + str(round(float(result_df['mae'].iloc[0]), 2))
# mse_test = 'MSE : ' + str(round(float(result_df['mse'].iloc[0]), 2))
# r_test = 'R2 : ' + str(round(float(result_df['r_square'].iloc[0]), 2))
# plt.text(0, -1.5, mae_test, fontsize=12)
# plt.text(0, -2, mse_test, fontsize=12)
# plt.text(0, -2.5, r_test, fontsize=12)

# plt.subplot(1, 2, 2)
# plt.scatter(logS_total, pred_logS_total, alpha=0.4)
# plt.plot(logS_total, logS_total, alpha=0.4, color='black')
# plt.xlabel("logS_total")
# plt.ylabel("pred_logS_total")

# plt.tight_layout()
# plt.savefig(os.path.join(args.output_path, args.exp_name + ".png"))
# print()