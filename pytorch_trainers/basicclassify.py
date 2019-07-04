import torch

def classify(model, x, y, loss_f, learning_rate = 1e-4, steps = 500):
    # Basic  classifcation, using torch in build optimisers. 
    optimiser = torch.optim.Adam(model.paramters, lr = learning_rate)

    for t in range(steps):
        y_pred = model(x)

        loss = loss_f(y_pred, y)

        optimser.zero_grad()

        loss.backward()

        optimser.step()
