import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
import glob
from PIL import Image

grit_label_dict = {'ATT': 0, 'GIVEUP': 1, 'GUESS': 2, 'NOTR': 3, 'SHINT': 4, 'SKIP': 5, 'SOF': 6}

class PadSequenceFull:
    def __call__(self, batch):
		# Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
		# Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
		# Also need to store the length of each sequence
		# This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
		# Don't forget to grab the labels of the *sorted* batch
        labels = torch.LongTensor([elt[2] for elt in sorted_batch])
        meta = torch.FloatTensor([[elt[1][i] for i in range(len(elt[1]))] for elt in sorted_batch])
        return sequences_padded, meta, lengths, labels

columns = ['timeToFirstAttempt', \
            'timeToFirstHint', \
            'timeToSolve', \
            'timeToSecondAttempt', 'timeToThirdAttempt', 'timeToSecondHint', \
            'timeToThirdHint'
            ]

class GritOpenFaceTiming(Dataset):
    def __init__(self, csv_path, early_sec=-1, nclass=7):
        self.annot = pd.read_csv(csv_path)
        # self.check_all()
        self.nclass = int(nclass)
        self.length = len(self.annot)
        print(self.length)
        
        self.early_sec = early_sec

    def check_all(self):
        print(len(self.annot))
        list_names = []
        for index, row in self.annot.iterrows():
            print(index)
            features_path = self.annot['af_features_path'].iloc[index]
            if not os.path.exists(features_path):
                print('Dropping ' + self.annot['af_features_path'].iloc[index])
                list_names.append(features_path)
        print(list_names)
        print('All good.')

    def __getitem__(self, index):
        feature_path = self.annot['openface_features_path'].iloc[index]
        features = torch.FloatTensor(np.load(feature_path))
        text_label = str(self.annot['effort'].iloc[index])
        label = grit_label_dict[text_label]

        if self.nclass == 2:
            if label in [4, 6]:
                label = 0
            else:
                label = 1 
       
        label = torch.LongTensor([label]).squeeze()

        deep_feature_path = self.annot['af_features_path'].iloc[index]
        deep_features = torch.FloatTensor(np.load(deep_feature_path)).squeeze()
        if features.size(0) == deep_features.size(0):
            features = torch.cat([features, deep_features], dim=1)
        elif features.size(0) > deep_features.size(0):
            features = torch.cat([features[:deep_features.size(0),:], deep_features], dim=1)
        else:
            features = torch.cat([features, deep_features[:features.size(0),:]], dim=1)

        if self.early_sec != -1:
            features = features[:int(self.early_sec)*3, :]        

        meta = []
        for c in columns:
            meta.append(self.annot[c].iloc[index])
        meta = torch.FloatTensor(meta)
        
        return features, meta, label

    def __len__(self):
        return self.length

class GritOpenFaceNext(Dataset):
    def __init__(self, csv_path):
        self.annot = pd.read_csv(csv_path)
        
        # self.check_all()

        self.length = len(self.annot)
        print(self.length)

    def check_all(self):
        print(len(self.annot))
        list_names = []
        list_indices = []
        for index, row in self.annot.iterrows():
            features_path = self.annot['af_features_path'].iloc[index]
            if not os.path.exists(features_path):
                print('Dropping ' + self.annot['af_features_path'].iloc[index])
                list_names.append(features_path)
                list_indices.append(index)
       	self.annot.drop(list_indices, inplace=True)
        print(list_names)
        print('All good.')

    def __getitem__(self, index):
        feature_path = self.annot['openface_features_path'].iloc[index]
        features = torch.FloatTensor(np.load(feature_path))
        text_label = str(self.annot['next_effort'].iloc[index])
        label = grit_label_dict[text_label]
        if label in [0, 4, 6]:
            label = 1
        else:
            label = 0
        label = torch.LongTensor([label]).squeeze()
        
        deep_feature_path = self.annot['af_features_path'].iloc[index]
        deep_feature_path = os.path.join(deep_feature_path.split('/')[0], "dual2-features", deep_feature_path.split('/')[-1])
        deep_features = torch.FloatTensor(np.load(deep_feature_path)).squeeze()
        if features.size(0) == deep_features.size(0):
            features = torch.cat([features, deep_features], dim=1)
        elif features.size(0) > deep_features.size(0):
            features = torch.cat([features[:deep_features.size(0),:], deep_features], dim=1)
        else:
            features = torch.cat([features, deep_features[:features.size(0),:]], dim=1)

        meta = torch.FloatTensor([self.annot['probDiff'].iloc[index], self.annot['mastery'].iloc[index], self.annot['next_probDiff'].iloc[index]])

        return features, meta, label

    def __len__(self):
        return self.length