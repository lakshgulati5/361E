
import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _format_number(x, pos=None):
    # If value is (close to) integer and >= 1000, use commas
    try:
        if float(x).is_integer() and abs(x) >= 1000:
            return f"{int(x):,}"
    except Exception:
        pass
    # Otherwise show exactly 4 decimal places
    return f"{x:.4f}"


def main(csv_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(csv_path):
        print('file not found at', csv_path)
        sys.exit(1)

    epochs = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            e, tl, tel, ta, tea = row
            epochs.append(int(e))
            train_losses.append(float(tl))
            test_losses.append(float(tel))
            train_accs.append(float(ta))
            test_accs.append(float(tea))

    # Loss plot (separate)
    loss_png = os.path.join(script_dir, 'loss_plot.png')
    fig1, ax1 = plt.subplots()
    l1, = ax1.plot(epochs, train_losses, marker='o', color='C0', label='Training Loss')
    l2, = ax1.plot(epochs, test_losses, marker='o', color='C1', label='Test Loss')
    ax1.set_title('Training and Test Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_format_number))
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig1.subplots_adjust(right=0.75)
    plt.tight_layout()
    fig1.savefig(loss_png)
    plt.close(fig1)
    print('Saved', loss_png)

    # Accuracy plot (separate)
    acc_png = os.path.join(script_dir, 'accuracy_plot.png')
    fig2, ax2 = plt.subplots()
    a1, = ax2.plot(epochs, train_accs, marker='s', linestyle='--', color='C2', label='Training Accuracy')
    a2, = ax2.plot(epochs, test_accs, marker='s', linestyle='--', color='C3', label='Test Accuracy')
    ax2.set_title('Training and Test Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_format_number))
    ax2.legend([a1, a2], [a1.get_label(), a2.get_label()], loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig2.subplots_adjust(right=0.75)
    plt.tight_layout()
    fig2.savefig(acc_png)
    plt.close(fig2)
    print('Saved', acc_png)


if __name__ == '__main__':
    csv_arg = sys.argv[1]
    main(csv_arg)
