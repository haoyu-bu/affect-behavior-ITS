import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import sklearn
from sklearn import metrics

import sys, os, argparse

sys.path.append('src/dataset')
sys.path.append('src/model')
import dataset
import rnn

# Device configuration
device = torch.device('cuda')

# Hyper-parameters
small_dim = 100
hidden_size = 200
num_layers = 2
num_classes = 2
batch_size = 1
num_epochs = 100
learning_rate = 3e-4
seed = 0
loss_lambda = 0.6

np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    for idx in range(5):
        train_csv = 'data/train_next_{}.csv'.format(idx)
        test_csv = 'data/test_next_{}.csv'.format(idx)
        output_log_path = './logs/coves_{}.txt'.format(idx)
        print(train_csv)
        print(test_csv)
        print(output_log_path)

        train_dataset = dataset.GritOpenFaceNext(csv_path=train_csv)

        test_dataset = dataset.GritOpenFaceNext(csv_path=test_csv)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                num_workers=10,
                                                shuffle=True,
                                                collate_fn=dataset.PadSequenceFull())

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1, 
                                                num_workers=10,
                                                shuffle=False,
                                                collate_fn=dataset.PadSequenceFull())

        model = rnn.RNNAttentionMultiloss(small_dim, hidden_size, num_layers, num_classes).to(device)


        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        total_step = len(train_loader)
        best_fscore = 0
        for epoch in range(1, num_epochs+1):
            train(num_epochs, epoch, total_step, train_loader, model, optimizer, criterion, device, idx)
            fscore = val(epoch+1, model, test_loader, device, output_log_path)
            if fscore > best_fscore:
                best_fscore = fscore
                print("save checkpoint.")
                torch.save(model.state_dict(), 'checkpoint/nextprob_best_{}.pth'.format(idx))
            if epoch == num_epochs:
                print("save checkpoint.")
                torch.save(model.state_dict(), 'checkpoint/nextprob_last_{}.pth'.format(idx))
            


def train(num_epochs, epoch, total_step, train_loader, model, optimizer, criterion, device, index):
    mean_loss = 0.0
    total = 0
    for i, (features, meta, lengths, labels) in enumerate(train_loader):
        facial_feats = features[:,:,:49].to(device)
        affect_feats = features[:,:,49:49 + 8192].to(device)
        meta_feats = meta.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        batch_size = features.size(0)
        total += batch_size
        # Forward pass
        outputs, outputs_m = model(facial_feats, affect_feats, meta_feats, lengths)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(outputs_m, labels)
        loss = (1 - loss_lambda) * loss1 + loss_lambda * loss2

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item() * batch_size
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, num_epochs, i+1, total_step, loss.item()))

    print('Mean loss: {}'.format(mean_loss / total))

def val(epoch, model, test_loader, device, output_log_path):
    # Test the model
    y_true = []
    y_pred = []
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (features, meta, lengths, labels) in enumerate(test_loader):
            facial_feats = features[:,:,:49].to(device)
            affect_feats = features[:,:,49:49 + 8192].to(device)
            meta_feats = meta.to(device)

            labels = labels.to(device)
            lengths = lengths.to(device)
            outputs, outputs_m = model(facial_feats, affect_feats, meta_feats, lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for idx in range(labels.size(0)):
                y_true.append(labels[idx].cpu().item())
                y_pred.append(predicted[idx].cpu().item())

        accuracy = 100 * correct / total
        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
        score_matrix = metrics.precision_recall_fscore_support(y_true, y_pred)
        print('Score matrix:')
        print(np.array_str(np.array(score_matrix), precision=2, suppress_small=True))
        print('Means of matrix:')
        print(np.mean(score_matrix, axis=1))
        print('*********************************** F-Score: ', np.mean(score_matrix, axis=1)[2])
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        # print(y_true, y_pred)
        print('Confusion matrix:')
        print(np.array_str(np.array(confusion_matrix), precision=2, suppress_small=True))
        print('CK:', metrics.cohen_kappa_score(y_true, y_pred))

        with open(output_log_path, 'a+') as f:
            f.write('****************************\n' + 
                    str(epoch-1)             + '\n' +
                    str(accuracy)          + '\n' +
                    str(np.array_str(np.array(score_matrix), precision=2, suppress_small=True)) + '\n' +
                    str(np.mean(score_matrix, axis=1)) + '\n' +
                    str(np.mean(score_matrix, axis=1)[2]) +'\n'           
                    )
        return np.mean(score_matrix, axis=1)[2]


if __name__ == '__main__':
    main()
