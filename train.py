import torch
from torch import nn
from itertools import chain
from model import sb_vae
from dataloader import get_mnist


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def train_model(model, train_loader, test_loader, num_epochs=500, learning_rate=5e-3):
    gd = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_results = []
    for _ in range(num_epochs):
        for i, (batch, _) in enumerate(train_loader):
            total = len(train_loader)
            gd.zero_grad()
            batch = batch.view(-1, D).to(device)
            nll, kl = model(batch)
            loss_value = nll + kl
            loss_value.backward()
            train_losses.append(loss_value.item())
            if (i + 1) % 10 == 0:
                print('\rTrain loss:', train_losses[-1],
                      'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)
            gd.step()
        test_loss = 0.
        for i, (batch, _) in enumerate(test_loader):
            batch = batch.view(-1, D).to(device)
            nll, kl = model(batch)
            batch_loss = nll + kl
            test_loss += (batch_loss - test_loss) / (i + 1)
        print('\nTest loss after an epoch: {}'.format(test_loss))

D = 28 * 28
model = sb_vae(D, 100, 0.5, 500, 0, 1)
model.to(device)
train_loader, test_loader = get_mnist()
train_model(model, train_loader, test_loader)
torch.save(model.state_dict(), "model.pt")
    
