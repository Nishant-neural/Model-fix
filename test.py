import torch
from training.train import train_model
from utility.plot import compare_histories
from models.mlp import SimpleMLP
from data.dataloader import get_dataloaders
from utility.diagnosis import diagnose 
from utility.explain import explain
from utility.suggestion import suggest
from auto import run_experiment

base_config = {
    "dropout": 0.0,
    "hidden_size": 256,
    "lr": 0.001,
    "epochs": 20,
    "optimizer": torch.optim.Adam,
    "criterion": torch.nn.CrossEntropyLoss()
}

train_loader, val_loader = get_dataloaders()
model = SimpleMLP(dropout_rate=0.0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20)         


problem = diagnose(history)

new_history, new_config = run_experiment(
    train_loader,
    val_loader,
    problem,
    base_config
)

compare_histories(history , new_history)

# print(f"\nDiagnosis: {problem}")        
# print(explain(problem))          
# print("\nSuggestions:")         

# for s in suggest(problem):       
    # print("-", s)