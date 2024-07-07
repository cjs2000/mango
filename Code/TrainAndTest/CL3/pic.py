import pandas as pd
import matplotlib.pyplot as plt


path = "checkpoints/shufflenet_eca_gc_4/shufflenet_eca_gc_4_training_results.xlsx"

#path2 = "checkpoints/shufflenet_eca_gc_4/shufflenet_eca_gc_4_training_results.xlsx"
# 从Excel文件读取数据
df = pd.read_excel(path)

# 获取所需列的数据
epochs = df['Epoch']
validation_accuracies = df['Validation_Accuracy']
validation_losses = df['Validation_Loss']

'''
df2 = pd.read_excel(path2)
# 获取所需列的数据
epochs2 = df2['Epoch']
validation_accuracies2 = df2['Validation_Accuracy']
validation_losses2 = df2['Validation_Loss']
'''



# 绘制第一张图像：Epoch vs Validation_Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_accuracies, label='shufflenet',marker=',', linestyle='-', color='#1f77b4')
#plt.plot(epochs, validation_accuracies2, label='shufflenet_eca_gc',marker=',', linestyle='-', color='#ff8519')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Epoch vs Validation Accuracy')
plt.grid(True)
plt.xticks(ticks=range(0, max(epochs)+1, 10))  # 设置横轴以10为单位分割
#plt.legend(loc="lower right")

plt.savefig(path+'epoch_vs_validation_accuracy.jpg', format='jpg', dpi=300)
plt.close()

# 绘制第二张图像：Epoch vs Validation_Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, validation_losses, marker=',', linestyle='-', color='#ff8519')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Epoch vs Validation Loss')
plt.grid(True)
plt.xticks(ticks=range(0, max(epochs)+1, 10))  # 设置横轴以10为单位分割
plt.savefig(path+'epoch_vs_validation_loss.jpg', format='jpg', dpi=300)
plt.close()

print("图像已保存为jpg格式。")
