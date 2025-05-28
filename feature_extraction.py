import torch
import numpy
from data_loader import *
from modeling import *

def extract_and_save_features (model, dataloader, path) :
    model.eval()
    features = []
    with torch.no_grad() :
        for batch_data, batch_labels in dataloader :
            out = model(batch_data)
            features = features.append(features, out)

    torch.save({'features': features, 'labels': batch_labels}, path)
    
    print('load and save success')
    return


cnn2d = CNN2D()
dataLoader2d = get2dDataLoader()
features2dPath = 'features/feature_2d.pt'
extract_and_save_features(cnn2d, dataLoader2d, features2dPath)