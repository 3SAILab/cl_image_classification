import torch

def train_loader(train_dataset,batch_size,shuffle=True,nw):
    return torch.utils.DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=nw)

def val_loader(val_dataset,batch_size,shuffle=True,nw):
    return torch.utils.DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=nw)