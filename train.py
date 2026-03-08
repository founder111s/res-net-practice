import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet import ResNet18
import matplotlib.pyplot as plt
import os

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=128, 
    shuffle=True,
    num_workers=4
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=128, 
    shuffle=False,
    num_workers=4
)

# 模型初始化
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(epochs=10):
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')
    
    # 保存模型
    torch.save(model.state_dict(), 'resnet_mnist.pth')
    
    # 绘制训练曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('results/training_curve.png')
    plt.close()
    
    return train_losses, train_accs, test_accs

# 测试函数
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

# 开始训练
if __name__ == "__main__":
    train_losses, train_accs, test_accs = train(epochs=10)