#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from nets import MappingNet
from toy_data import generate_data


def main():
    shuffle=False
    theta = np.pi / 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

    # create toy data as a tensor
    origin_space, target_space = generate_data(rot_matrix, sample_size=25000, plot=True, shuffle=shuffle)
    data = np.hstack((origin_space, target_space))
    data_tensor = torch.from_numpy(data).float()

    # create neural net, define optimizer and loss criterion
    net = MappingNet()
    optimizer = torch.optim.Adam(params=net.parameters())
    criterion = torch.nn.MSELoss()

    # begin training loops
    iteration = 0
    loss_curve = []
    t0 = time.time()
    for row in data_tensor:
        iteration += 1

        # get input and target
        origin = row[:2]
        target = row[2:]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net(origin)
        loss = criterion(outputs, target)

        # backward pass
        loss.backward()

        # optimize
        optimizer.step()

        # print loss every 1000 samples
        if iteration % 1000 == 0:
            print(f'[iter {iteration}] loss: {loss.data.item()}')

        # add loss to a list to be plotted
        loss_curve.append(loss.data.item())

    t1 = time.time()
    print(f'Time taken to finish: {t1 - t0}')

    # plot loss curve
    plt.clf()
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.show()

    # generate test data and make predictions
    test_origin, test_target = generate_data(rot_matrix, sample_size=10000, plot=False)
    test_tensor = torch.from_numpy(test_origin).float()
    test_prediction = net(test_tensor).detach().numpy()

    # plot test data and its transformation
    plt.clf()
    plt.scatter(x=test_origin[:, 0], y=test_origin[:, 1], c='red')
    plt.scatter(x=test_prediction[:, 0], y=test_prediction[:, 1], c='blue')
    plt.axis([-5, 5, -5, 5])
    plt.show()


if __name__ == '__main__':
    main()
