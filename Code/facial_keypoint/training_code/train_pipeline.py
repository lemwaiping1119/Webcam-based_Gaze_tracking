import torch
import torch.optim as optim
import torch.nn as nn
import os

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, device=torch.device("cpu"), save_interval=10, save_dir=""):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, keypoints in train_loader:
            imgs = imgs.to(device)
            keypoints = keypoints.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, keypoints in val_loader:
                imgs = imgs.to(device)
                keypoints = keypoints.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, keypoints)
                val_running_loss += loss.item() * imgs.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint every save_interval epochs.
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    return model
