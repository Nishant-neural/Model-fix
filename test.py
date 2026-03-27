import torch
from training.train import train_model
from utility.plot import plot_history
from models.mlp import SimpleMLP
from data.dataloader import get_dataloaders
from utility.diagnosis import diagnose 
from utility.explain import explain
from utility.suggestion import suggest

train_loader, val_loader = get_dataloaders()
model = SimpleMLP(dropout_rate=0.0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20)

plot_history(history)
result = diagnose(history)

print(f"\nDiagnosis: {result}")
print(explain(result))
print("\nSuggestions:")

for s in suggest(result):
    print("-", s)