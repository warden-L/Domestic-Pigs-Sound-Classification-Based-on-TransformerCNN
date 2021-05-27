import numpy as np
import os
import warnings
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class model(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2])

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=145,
            nhead=5,
            dim_feedforward=512,
            dropout=0.5,
        )

        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.conv2Dblock2 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.5),
        )
        self.fc1_linear = nn.Linear(657, num_class)

        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)
        x_maxpool = self.transformer_maxpool(x)
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
        x = x_maxpool_reduced.permute(2, 0, 1)
        transformer_output = self.transformer_encoder(x)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        complete_embedding = torch.cat([conv2d_embedding2, transformer_embedding], dim=1)
        output_logits = self.fc1_linear(complete_embedding)
        output_softmax = self.softmax_out(output_logits)
        return output_logits, output_softmax

file_name = 'MLMC'
train_x, train_y, val_x, val_y, test_x, test_y = np.load('./'+file_name+'/train_'+file_name+'.npy'), np.load('./'+file_name+'/train_label.npy'), np.load(
    './'+file_name+'/test_'+file_name+'.npy'), np.load('./'+file_name+'/test_label.npy'), np.load('./'+file_name+'/val_'+file_name+'.npy'), np.load('./'+file_name+'/val_label.npy')

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
class_weights[1] = class_weights[1]*1.6
class_weights = torch.FloatTensor(class_weights)

test_y_ =test_y

train_x = np.array([x.reshape(1, x.shape[0], x.shape[1]) for x in train_x])
test_x = np.array([x.reshape(1, x.shape[0], x.shape[1]) for x in test_x])
val_x = np.array([x.reshape(1, x.shape[0], x.shape[1]) for x in val_x])

train_x, train_y = torch.FloatTensor(train_x), torch.LongTensor(train_y)
val_x, val_y = torch.FloatTensor(val_x), torch.LongTensor(val_y)
test_x, test_y = torch.FloatTensor(test_x), torch.LongTensor(test_y)

train_data = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))

train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=0)

Epoch = 100000000
LR = 1e-3
early_epoch = 100
new_early_epoch = 0
new_val_loss = 0.0

net = model(4).to(device)

optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
for epoch in range(Epoch):
    print("\nEpoch %d" % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in enumerate(train_loader, 0):
        length = len(train_loader)
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

        optimizer.zero_grad()

        outputs, _ = net(inputs)
        loss = criterion(outputs.cpu(), F.one_hot(labels).float().cpu())
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()
    print('[epoch:%d] Loss: %.04f | Acc: %.8f%% '
          % (epoch + 1, sum_loss / (i + 1), 100.0 * correct / total))

    val_loss = 0.0
    val_correct = 0.0
    val_total = 0.0

    net.eval()
    with torch.no_grad():
        inputs, labels = Variable(val_x).to(device), Variable(val_y).to(device)
        outputs, _ = net(inputs)
        val_loss = criterion(outputs.cpu(), F.one_hot(labels).float().cpu())
        scheduler.step(val_loss)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels.data).cpu().sum().item()
    print('Val_Loss: %.04f | Val_Acc: %.8f%% '
          % (val_loss, 100.0 * val_correct / val_total))

    if (100.0 * val_correct / val_total <= new_val_loss):
        new_early_epoch += 1
    else:
        new_val_loss = 100.0 * val_correct / val_total
        new_early_epoch = 0
    if (new_early_epoch >= early_epoch):
        break


def per_class_accuracy(y_preds, y_true, class_labels=['0', '1', '2', '3']):
    return [np.mean([
        (y_true[pred_idx] == np.round(y_pred)) for pred_idx, y_pred in enumerate(y_preds)
        if y_true[pred_idx] == int(class_label)
    ]) for class_label in class_labels]


net.eval()
with torch.no_grad():
    inputs, labels = Variable(test_x).to(device), Variable(test_y).to(device)
    outputs, _ = net(inputs)
    loss = criterion(outputs.cpu(), F.one_hot(labels).float().cpu())
    scheduler.step(val_loss)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels.data).cpu().sum().item()

print('Test_Loss: %.04f | Test_Acc: %.8f%% '
      % (loss, 100.0 * correct / total))
print("per class acc:", per_class_accuracy(predicted.cpu().numpy(), labels.cpu().numpy()),
      np.mean(per_class_accuracy(predicted.cpu().numpy(), labels.cpu().numpy())))
f = int(100.0 * correct / total)
torch.save(net, './model/' + str(f) + '_tf_cnn_model.pkl')


from sklearn.metrics import precision_score,f1_score

precision = precision_score(test_y_, predicted, average='weighted')
recall = recall_score(test_y_, predicted, average='weighted')

f1_score = f1_score(test_y_, predicted, average='weighted')
accuracy_score_test = accuracy_score(test_y_, predicted)
print("Precision_score:",precision)
print("Recall_score:",recall)
print("F1_score:",f1_score)
print("Accuracy_score:",accuracy_score_test)

import matplotlib.pyplot as plt
from keras.utils import to_categorical
test_y = to_categorical(test_y_,num_classes=4)
test_auc=roc_auc_score(test_y,outputs)
print("test auc is : {:.4f}".format(test_auc))

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

emotions_dict = dict()
emotions_dict['feed']=' feed'
emotions_dict['normal']='normal'
emotions_dict['anxious']='anxious'
emotions_dict['howl']='howl'

conf_matrix = confusion_matrix(test_y_, predicted)
conf_matrix_norm = confusion_matrix(test_y_, predicted,normalize='true')

emotion_names = [emotion for emotion in emotions_dict.values()]

confmatrix_df = pd.DataFrame(conf_matrix, index=emotion_names, columns=emotion_names)
confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=emotion_names, columns=emotion_names)

plt.figure(figsize=(16,6))
sn.set(font_scale=1.8)
plt.subplot(1,2,1)
plt.title('Confusion Matrix')
sn.heatmap(confmatrix_df, annot=True, annot_kws={"size": 18})
plt.subplot(1,2,2)
plt.title('Normalized Confusion Matrix')
sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={"size": 13})
plt.savefig('roc.jpg')
plt.show()
