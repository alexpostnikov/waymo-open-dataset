import torch
from tqdm.auto import tqdm

from tutorial.test_train import get_ade_from_pred_speed_with_mask, get_speed_ade_with_mask


def train(model, loader, optimizer, num_ep=10):
    losses = torch.rand(0)
    for epoch in range(num_ep):  # loop over the dataset multiple times
        pbar = tqdm(loader)
        for chank, data in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(data)

            loss = get_ade_from_pred_speed_with_mask(data, outputs).mean()
            # loss = ade_loss(data, outputs)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                speed_ade = get_speed_ade_with_mask(data, outputs.clone())
                lin_ade = get_ade_from_pred_speed_with_mask(data, outputs.clone())
                losses = torch.cat([losses, torch.tensor([loss.detach().item()])], 0)
                pbar.set_description("ep %s chank %s" % (epoch, chank))
                pbar.set_postfix({"loss": losses.mean().item(),
                                  "median": speed_ade.median().item(),
                                  "max": speed_ade.max().item(),
                                  "lin_ade": lin_ade.mean().item()})
                if len(losses) > 500:
                    losses = losses[100:]