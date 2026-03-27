def explain(result):
    if result == "overfitting":
        return "Model is overfitting: training loss is much lower than validation loss."
    
    elif result == "underfitting":
        return "Model is underfitting: it is not learning enough patterns."
    
    else:
        return "Model is learning well and generalizing."