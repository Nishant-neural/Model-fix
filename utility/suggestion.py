def suggest(problem):
    
    if problem == "overfitting":
        return [
            "Add dropout (e.g., 0.3 - 0.5)",
            "Reduce model size (fewer neurons)",
            "Use L2 regularization (weight decay)",
            "Train for fewer epochs (early stopping)"
        ]
    
    elif problem == "underfitting":
        return [
            "Increase model size (more neurons)",
            "Train longer (more epochs)",
            "Reduce regularization",
            "Use better features / architecture"
        ]
    
    elif problem == "good_fit":
        return [
            "Model is performing well",
            "You can try small optimizations",
            "Consider saving this model"
        ]
    
    else:
        return ["Not enough information"]