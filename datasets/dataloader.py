import torch

def train_loader(train_dataset, batch_size, nw, shuffle=True):
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=nw)

def val_loader(val_dataset,batch_size,nw,shuffle=False):
    return torch.utils.data.DataLoader(val_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=nw)