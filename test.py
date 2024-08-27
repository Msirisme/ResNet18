import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import ResNet18, Residual
from torchvision.datasets import ImageFolder
from PIL import Image
def test_data_process():
    ROOT_TRAIN = r'data\test'
    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155,  0.06216329, 0.05930814])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader

def test_model_process(model, test_dataloader):
    device ="cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    test_correct= 0.0
    test_num=0
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output=model(test_data_x)
            pre_lab=torch.argmax(output,dim=1)
            test_correct+=torch.sum(pre_lab==test_data_y.data)
            test_num +=test_data_x.size(0)
    test_acc = test_correct.double().item() / test_num
    print("测试的准确率为：", test_acc)

if __name__ == "__main__":
    model = ResNet18(Residual)
    model.load_state_dict(torch.load('best_model.pth'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    classes = ['戴口罩', '不带口罩']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为验证模型
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：",  classes[result], "------", "真实值：", classes[label])
    image = Image.open('no_mask.jfif')

    normalize = transforms.Normalize([0.17263485, 0.15147247, 0.14267451], [0.0736155, 0.06216329, 0.05930814])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测值：", classes[result])