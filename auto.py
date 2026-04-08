from models.mlp import SimpleMLP
from training.train import train_model

def run_experiment(train_loader, val_loader, problem, base_config):
    
    config = base_config.copy()

  
    if problem == "overfitting":
        config["dropout"] = 0.5
    
    elif problem == "underfitting":
        config["hidden_size"] = 512


    model = SimpleMLP(dropout_rate=config["dropout"], hidden_size=config["hidden_size"])

    optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
    criterion = config["criterion"]

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=config["epochs"]
    )

    return history, config