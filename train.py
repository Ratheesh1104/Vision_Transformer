from dataset import CIFAR10DataModule
from vit import VisionTransformer
from trainer import Trainer
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = CIFAR10DataModule(batch_size = 32)
    data.setup()

    model = VisionTransformer(num_classes=10)

    trainer = Trainer(model, device)

    for epoch in range(5):
        loss = trainer.train_epoch(data.train_dataloader())
        acc = trainer.evaluate(data.test_dataloader)

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

if __name__ == "__main__":
    main()