from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
from models.inceptionv3 import *
from models.resnet import *
from models.eca_resnet import *
from models.Vit_transformer import *
from models.mobilenet import *
from models.mobilenetv2 import *
from models.shufflenet import *
from models.shufflenetv2 import *
from models.squeezenet import *
from models.SwinTransformer import *
from models.densenet import *
from models.shufflenet_eca import *
from shufflenet_IM.shufflenet_SE import *
from shufflenet_IM.shufflenet_BAM import *
from shufflenet_IM.shufflenet_EMA import *
from shufflenet_IM.shufflenet_MobileViTAttention import *
from shufflenet_IM.shufflenet_GAM import *
from shufflenet_IM.shufflenet_ECA_GC import *
from models.LeNet5 import *
from models.AlexNet import *
from models.EfficientNet import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
data_path = "data_split" # todo 数据集路径

# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    # 'model': 'vit_tiny_patch16_224',  # 选择预训练模型
    # 'model': 'resnet50d',  # 选择预训练模型
    'model': 'EfficientNet',  # 选择预训练模型
    "img_size": 224,  # 图片输入大小
    "train_dir": osp.join(data_path, "train"),  # todo 训练集路径
    "val_dir": osp.join(data_path, "val"),  # todo 验证集路径
    'device': device,  # 设备
    'lr': 1e-3,  # 学习率
    'batch_size': 4,  # 批次大小
    'num_workers': 8,  # 进程
    'epochs': 150,  # 轮数
    "save_dir": "checkpoints/",  # todo 保存路径
    "pretrained": False,
     "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    'weight_decay': 1e-5  # 学习率衰减
}


# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)  # 从预训练的库中加载模型
        # self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path="pretrained/resnet50d_ra2-464e36ba.pth")  # 从预训练的库中加载模型
        # classifier
        if model_name[:3] == "res":
            n_features = self.model.fc.in_features  # 修改全连接层数目
            self.model.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features  # 修改全连接层数目
            self.model.head = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)
        # resnet修改最后的全链接层
        print(self.model)  # 返回模型

    def forward(self, x):  # 前向传播
        x = self.model(x)
        return x


# 定义训练流程
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):  # 开始训练
        images = images.to(params['device'], non_blocking=True)  # 加载数据
        target = target.to(params['device'], non_blocking=True)  # 加载模型
        output = model(images)  # 数据送入模型进行前向传播
        loss = criterion(output, target.long())  # 计算损失
        f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
        recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
        acc = accuracy(output, target)  # 计算准确率分数
        metric_monitor.update('Loss', loss.item())  # 更新损失
        metric_monitor.update('F1', f1_macro)  # 更新f1
        metric_monitor.update('Recall', recall_macro)  # 更新recall
        metric_monitor.update('Accuracy', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新优化器
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]  # 返回结果


# 定义验证流程
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            loss = criterion(output, target.long())  # 计算损失
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]


# 展示训练过程的曲线
def show_loss_acc(acc, loss, val_acc, val_loss,save_dir):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # 保存在savedir目录下。
    save_path = osp.join(save_dir, "results.png")
    plt.savefig(save_path, dpi=100)



# 初始化TP、FP、FN的计数器
num_classes = 4
TP = np.zeros(num_classes)
FP = np.zeros(num_classes)
FN = np.zeros(num_classes)




def show_heatmaps(title, x_labels, y_labels, harvest,save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)


def test(val_loader, model, params, class_names,model_name,save_dir):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    # 初始化计数器
    ac = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            a = int(target.item())

            output = model(images)  # 前向传播

            label_id = int(torch.argmax(output).item())
            b = label_id
            if a == b:
                TP[a] += 1
            else:
                FP[b] += 1
                FN[a] += 1

            if a == b:
                correct_predictions += 1
            total_samples += 1


            #loss = criterion(output, target.long())  # 计算损失
            # print(output)

            target_numpy = target.cpu().numpy()
            y_pred = torch.softmax(output, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            test_real_labels.extend(target_numpy)
            test_pre_labels.extend(y_pred)

            # print(target_numpy)
            # print(y_pred)
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc

            # metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)

            stream.set_description(
                "mode: {epoch}.  {metric_monitor}".format(
                    epoch="test",
                    metric_monitor=metric_monitor)
            )

        # 计算准确率
        ac = correct_predictions / total_samples
        # 打印结果
        print(f"Accuracy: {ac:.5f}")
        ac = f"Accuracy: {ac:.5f}"

    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    # print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    # print()
    heat_maps_float = heat_maps / heat_maps_sum
    # print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name=save_dir+"/heatmap_{}.png".format(model_name))
    # 加上模型名称


    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['F1']["avg"], \
           metric_monitor.metrics['Recall']["avg"], ac


def te(model_name,model_path,save_path,model):
    paratest = {
        # 'model': 'vit_tiny_patch16_224',  # 选择预训练模型
        # 'model': 'efficientnet_b3a',  # 选择预训练模型
        'model': model_name,  # 选择预训练模型
        "img_size": 224,  # 图片输入大小
        "test_dir": osp.join(data_path, "test"),
        'device': device,  # 设备
        'batch_size': 1,  # 批次大小
        'num_workers': 0,  # 进程
        "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    }


    data_transforms = get_torch_transforms(img_size=paratest["img_size"])  # 获取图像预处理方式
    # train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    # valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    # print(valid_dataset)
    test_dataset = datasets.ImageFolder(paratest["test_dir"], valid_transforms)
    class_names = test_dataset.classes
    print(class_names)
    # valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    test_loader = DataLoader(  # 按照批次加载训练集
        test_dataset, batch_size=paratest['batch_size'], shuffle=True,
        num_workers=paratest['num_workers'], pin_memory=True,
    )
    #model = shufflenet()  # todo 模型选择

    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    # 指标上的测试结果包含三个方面，分别是acc f1 和 recall, 除此之外，应该还有相应的热力图输出，整体会比较好看一些。
    criterion = nn.CrossEntropyLoss().to(paratest['device'])  # 设置损失函数
    acc, f1, recall,ac = test(test_loader, model, paratest, class_names,model_name,save_path)

    # 计算精确率、召回率和F1分数
    precision = np.zeros(num_classes)
    rc = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        if TP[i] + FP[i] > 0:
            precision[i] = TP[i] / (TP[i] + FP[i])
        else:
            precision[i] = 0.0

        if TP[i] + FN[i] > 0:
            rc[i] = TP[i] / (TP[i] + FN[i])
        else:
            rc[i] = 0.0

        if precision[i] + rc[i] > 0:
            f1_score[i] = 2 * (precision[i] * rc[i]) / (precision[i] + rc[i])
        else:
            f1_score[i] = 0.0

    save_path = save_path + "/test.txt"

    # 打开文件以写入
    with open(save_path, "w") as file:
        # 写入准确率
        file.write(ac + "\n")

        # 输出结果并保存到文件
        for i in range(num_classes):
            output = f"Class {class_names[i]} - Precision: {precision[i]:.4f}, Recall: {rc[i]:.4f}, F1 Score: {f1_score[i]:.4f}\n"
            print(output.strip())  # 打印到控制台
            file.write(output)  # 写入文件






if __name__ == '__main__':

    accs = []
    losss = []
    val_accs = []
    val_losss = []
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)  # 加载训练集
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    if params['pretrained'] == True:
        save_dir = osp.join(params['save_dir'], params['model'])  # 设置模型保存路径
    else:
        save_dir = osp.join(params['save_dir'], params['model'])  # 设置模型保存路径
    if not osp.isdir(save_dir):  # 如果保存路径不存在的话就创建
        os.makedirs(save_dir)  #
        print("save dir {} created".format(save_dir))
    train_loader = DataLoader(  # 按照批次加载训练集
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )
    print(train_dataset.classes)

    '''
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],
                 pretrained=params['pretrained']) # 加载模型    todo 模型选择
    '''

    model = efficientnet_b0()           # todo 模型选择

    model_t = model
    print(model)


    # model = nn.DataParallel(model)  # 模型并行化，提高模型的速度
    # resnet50d_1epochs_accuracy0.50424_weights.pth
    model = model.to(params['device'])  # 模型部署到设备上
    criterion = nn.CrossEntropyLoss().to(params['device'])  # 设置损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  # 设置优化器
    # 损失函数和优化器可以自行设置修改。
    # criterion = nn.CrossEntropyLoss().to(params['device'])  # 设置损失函数
    # optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  # 设置优化器
    best_acc = 0.0  # 记录最好的准确率
    # 只保存最好的那个模型。
    i = int(params['epochs'] + 1)
    for epoch in range(1, params['epochs'] + 1):  # 开始训练
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        if val_acc >= best_acc:
            # 保存的时候设置一个保存的间隔，或者就按照目前的情况，如果前面的比后面的效果好，就保存一下。
            # 按照间隔保存的话得不到最好的模型。
            save_path = osp.join(save_dir, f"best.pth")
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc

        save_path = osp.join(save_dir, f"last.pth")
        torch.save(model.state_dict(), save_path)



            
    show_loss_acc(accs, losss, val_accs, val_losss, save_dir)

    # 创建一个包含这些数据的 DataFrame
    df = pd.DataFrame({
        'Epoch': range(1, len(accs) + 1),
        'Train_Accuracy': accs,
        'Train_Loss': losss,
        'Validation_Accuracy': val_accs,
        'Validation_Loss': val_losss
    })
    sp = osp.join(save_dir, f"{params['model']}_training_results.xlsx")
    # 保存 DataFrame 到 Excel 文件
    df.to_excel(sp, index=False)
    print("训练已完成，模型和训练日志保存在: {}".format(save_dir))

    save_dir = osp.join(params['save_dir'], params['model'])
    model_path = osp.join(save_dir, f"best.pth")
    te(params['model'], model_path,save_dir,model_t)



