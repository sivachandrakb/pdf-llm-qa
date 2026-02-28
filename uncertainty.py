import torch

def calculate_entropy(scores):
    probs = torch.nn.functional.softmax(scores[-1], dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))
    return entropy.item()

def confidence_from_entropy(entropy):
    return 1 / (1 + entropy)
