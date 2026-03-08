import torch
import torchvision
import torchvision.transforms as transforms
from models.resnet import ResNet18
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=128, 
    shuffle=False
)

# 加载模型
model = ResNet18().to(device)
model.load_state_dict(torch.load('resnet_mnist.pth'))
model.eval()

# 测试函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    return correct, total

# 可视化部分
def visualize_predictions():
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    plt.figure(figsize=(15, 10))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f'True: {labels[i].item()}, Pred: {predicted[i].item()}')
        plt.axis('off')
    plt.savefig('results/predictions.png')
    plt.close()

# 执行测试
if __name__ == "__main__":
    correct, total = test()
    visualize_predictions()