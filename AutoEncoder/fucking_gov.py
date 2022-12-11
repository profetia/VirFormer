import torch
import torch.nn as nn

torch.manual_seed(1111)
device = torch.device("cuda")
classifier_model = nn.Sequential(
        nn.Linear(250 * 200, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.ReLU()
    )
classifier_model = classifier_model.to(device)
criterion = nn.MSELoss()


optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)
classifier_model.train()
for index in range(1000):
    output = torch.randn(20, 250 * 200).to(device)
    label = torch.randn(20).to(device)
    result = classifier_model(output).squeeze()
    loss = criterion(result, label)
    loss.backward()
    optimizer.step()
    print(loss)
    print(optimizer.param_groups[0]['params'][0].grad.sum())
