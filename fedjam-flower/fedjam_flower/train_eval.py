from collections import OrderedDict
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict


def train(model, trainloader, epochs, device, lr=5e-5, is_lora=False, is_timm=False):
    """Train the model on the training set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            if is_timm:
                outputs = model(images)
            else:
                outputs = model(pixel_values=images).logits
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(model, testloader, device, is_lora=False, is_timm=False):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            if is_timm:
                outputs = model(images)
            else:
                outputs = model(pixel_values=images).logits
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(model, is_lora=False):
    if is_lora:
        state_dict = get_peft_model_state_dict(model)
    else:
        state_dict = model.state_dict()
    return [val.cpu().numpy() for _, val in state_dict.items()]


def set_weights(model, parameters, is_lora=False):
    if is_lora:
        peft_state_dict_keys = get_peft_model_state_dict(model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(model, state_dict)
    else:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
