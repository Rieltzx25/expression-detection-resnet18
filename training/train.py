import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # ==== Step 1: Data Transform & Augmentasi ====
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ==== Step 2: Data Loader ====
    train_dir = 'archive/train'
    test_dir = 'archive/test'

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")


    # ==== Step 3: Device (CPU/GPU) ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ==== Step 4: Transfer Learning Model (ResNet18) ====
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Ubah input layer ke 1 channel (grayscale)
    # The original resnet.conv1 has out_channels=64
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Ganti output layer sesuai jumlah kelas
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)

    # MODIFICATION: Fine-tune all layers for potentially better adaptation
    print("Fine-tuning all layers of the model.")
    for param in resnet.parameters():
        param.requires_grad = True

    # ==== Step 5: Loss, Optimizer, Scheduler ====
    criterion = nn.CrossEntropyLoss()
    
    # MODIFICATION: Potentially use a smaller learning rate when fine-tuning all layers.
    # Start with a rate like 1e-4 or 5e-5 and adjust if needed.
    # The original 0.0005 might be too high if all layers are trainable.
    optimizer = optim.Adam(resnet.parameters(), lr=0.0001) # Adjusted learning rate
    
    # MODIFICATION: Added verbose=True to scheduler for better logging
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=0.5, verbose=True)

    # ==== Step 6: Training Loop ====
    epochs = 60 # You might need to increase this depending on convergence
    
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    print("\nStarting training...")
    for epoch in range(epochs):
        resnet.train()
        running_loss, correct, total = 0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader) : # Print progress
                 print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item():.4f}')


        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct / total
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc)

        # Validation
        resnet.eval()
        val_running_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet(imgs)
                loss = criterion(outputs, labels) # Calculate validation loss
                val_running_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(test_dataset)
        epoch_val_acc = val_correct / val_total
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)
        
        scheduler.step(epoch_val_acc) # Step scheduler based on validation accuracy
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ==== Step 7: Save Model ====
    model_save_path = "expression_resnet18_pytorch_finetuned.pth"
    torch.save(resnet.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # ==== Step 8: Plot Accuracy and Loss ====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.title("Accuracy Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.title("Loss Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()