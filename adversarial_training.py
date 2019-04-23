import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from nets import MappingNet
from nets import DiscriminatorNet
from toy_data import generate_data


def training_step(net, discriminator, optimizer, loss_criterion, point, source):
    '''
    Performs the training step on either the mapper or the discriminator.

    Parameters:
        net: str -> 'mapper' or 'discriminator'
        discriminator: DiscriminatorNet -> the discriminator neural network
        optimizer: torch.optim object -> optimizer for the neural network
                                         being trained
        loss_criterion: torch.nn.modules.loss object -> loss function
        point: torch.Tensor -> data vector of shape (1, 2)
        source: torch.Tensor -> vector of shape (1, 2) describing the source of
                                the point, either a mapped point or a true
                                target point
    '''

    # zero grad the optimizer 
    optimizer.zero_grad()

    # get discriminator output for the given point
    discrim_output = discriminator(point)

    # calculate loss, the mapper tries to make the output incorrect and the
    # discriminator tries to make the output match the source
    if net == 'mapper':
        loss = loss_criterion(discrim_output, 1 - source)
    elif net == 'discriminator':
        loss = loss_criterion(discrim_output, source)
    else:
        raise Exception(f'Invalid parameter for "net": {net}')

    # backward propagate
    loss.backward(retain_graph=True)

    # optimizer step
    optimizer.step()


def main():
    # use gpu if cuda is available
    # note: cpu actually runs faster for simple architectures so we always
    # use cpu for now
    if torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # create toy data as a tensor
    origin_space, target_space = generate_data(sample_size=10000, plot=True)
    data = np.hstack((origin_space, target_space))
    data_tensor = torch.tensor(data=data,
                               dtype=torch.float,
                               device=device)

    # create neural nets, define optimizers and loss criterion
    map_net = MappingNet().to(device)
    discrim_net = DiscriminatorNet().to(device)
    map_optimizer = torch.optim.SGD(params=map_net.parameters(),
                                    lr=0.001,
                                    momentum=0.5)
    discrim_optimizer = torch.optim.SGD(params=discrim_net.parameters(),
                                        lr=0.001,
                                        momentum=0.5)
    criterion = torch.nn.BCELoss()

    # begin training loops
    t0 = time.time()
    for epoch in range(5):

        iteration = 0
        running_error = 0
        for row in data_tensor:
            iteration += 1

            # get input and target
            origin = row[:2].view(1, 2)
            target = row[2:].view(1, 2)

            # define source tensors
            origin_source = torch.tensor(data=[[1., 0.]]).to(device)
            target_source = torch.tensor(data=[[0., 1.]]).to(device)

            # train each net on the true target point
            training_step(net='mapper',
                          discriminator=discrim_net,
                          optimizer=map_optimizer,
                          loss_criterion=criterion,
                          point=target,
                          source=target_source)
            training_step(net='discriminator',
                          discriminator=discrim_net,
                          optimizer=discrim_optimizer,
                          loss_criterion=criterion,
                          point=target,
                          source=target_source)

            # train each net on a point mapped from the origin space to the target
            mapped_point = map_net(origin)
            training_step(net='mapper',
                          discriminator=discrim_net,
                          optimizer=map_optimizer,
                          loss_criterion=criterion,
                          point=mapped_point,
                          source=origin_source)
            training_step(net='discriminator',
                          discriminator=discrim_net,
                          optimizer=discrim_optimizer,
                          loss_criterion=criterion,
                          point=mapped_point,
                          source=origin_source)

            if iteration % 1000 == 0: 
                print(f'[{epoch} {iteration}] done')
                running_error = 0


    # Manually create the rotation matrix used, as a check
    # TODO !! do validation in an unsupervised way
    theta = np.pi / 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                           [np.sin(theta), np.cos(theta)]])

    # Print information from the training
    t1 = time.time()
    print(f'Time taken to finish: {t1 - t0}')

    print(f'True Rotation:\n{rot_matrix}')
    print(f'Learned Rotation:\n{map_net.fc1.weight.detach().numpy()}')

    # generate test data and make predictions
    test_origin, test_target = generate_data(sample_size=1000, plot=False)
    rotated_origin = test_origin @ rot_matrix
    test_tensor = torch.from_numpy(test_origin).float().to(device)
    test_prediction = map_net(test_tensor).cpu().detach().numpy()

    # plot test data, its transformation, and the true target space
    plt.clf()
    plt.scatter(x=test_origin[:, 0], y=test_origin[:, 1],
                c='red', label='origin space')
    plt.scatter(x=rotated_origin[:, 0], y=rotated_origin[:, 1],
                c='blue', label='target space')
    plt.scatter(x=test_prediction[:, 0], y=test_prediction[:, 1],
                c='yellow', label='transformed data')
    plt.axis([-5, 5, -5, 5])
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()