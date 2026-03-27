
import numpy as np

def diagnose(history):
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

 
    final_train = train_losses[-1]
    final_val = val_losses[-1]

    gap = final_val - final_train

    val_trend = val_losses[-5:]  
    is_val_increasing = val_trend[-1] > val_trend[0]

    train_trend = train_losses[-5:]
    is_train_decreasing = train_trend[-1] < train_trend[0]
    print("Val trend:", val_losses[-5:])

    if is_train_decreasing and is_val_increasing:
        return "overfitting"

    elif final_train > 1.0 and final_val > 1.0:
        return "underfitting"

    elif gap < 0.3:
        return "good_fit"

    else:
        return "uncertain"
    