import torch 

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    
    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            preds = model(x)
            loss = criterion(preds, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")

    return history