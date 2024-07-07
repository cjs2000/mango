
from PIL import Image
from torchutils import get_torch_transforms
from model.resnet import *





def predict_single():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    classes_names = ['Ao Mango', 'GuiQi Mango', 'Jinhuang Mango', 'Tainong Mango']
    image_path = r"D:\Code\Detect_Django\img\s.jpg"
    weights_path = r"D:\Code\Detect_Django\model\model.pth"
    data_transforms = get_torch_transforms(img_size=224)
    # train_transforms = data_transforms['train']
    valid_transforms = data_transforms['val']

    model = resnet18()  # todo 模型选择
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # 读取图片
    img = Image.open(image_path)
    img = valid_transforms(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    print(output)

    label_id = torch.argmax(output).item()
    # 获取预测类别的置信度
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence = str(probabilities[0, label_id].item())

    print("置信度："+confidence)

    predict_name = classes_names[label_id]

    id = str(label_id)
    predict_name = classes_names[label_id]
    print(f"{image_path}'s result is {predict_name}")
    return id


if __name__ == '__main__':

    # 单张图片预测函数
    id = predict_single()
    print(id)

