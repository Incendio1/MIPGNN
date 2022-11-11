import torch.nn.functional as F
import torch


def train(model, optimizer, data, beta):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # x_bar, out = model(data)
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + beta * F.mse_loss(x_bar, data.x)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        # _, logits = model(data)
        logits = model(data)
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
    return outs
