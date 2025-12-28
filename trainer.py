import torch.nn as nn
import torch.optim as optim
import torch


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr =3e-4
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
    
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0 ,0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total