

import matplotlib.pyplot as plt

def plot_history(history):
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.show()

def compare_histories(h1, h2):

    plt.plot(h1["val_loss"], label="before")
    plt.plot(h2["val_loss"], label="after")
    
    plt.legend()
    plt.title("Before vs After Fix")
    plt.show()