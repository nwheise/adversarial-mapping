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
        loss = 2*loss_criterion(discrim_output, source)
    else:
        raise Exception(f'Invalid parameter for "net": {net}')

    # backward propagate
    loss.backward(retain_graph=True)

    # optimizer step
    optimizer.step()

    return loss.data.item()


def weights_init_uniform(m):
    '''
    Initialization function to be applied to a neural net. Initializes
    weights to a sample from a unifrom distribution from
    [- 1 / sqrt(n), 1 / sqrt(n)] where n is the number of input features.
    '''

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        bound = 1 / np.sqrt(m.in_features)
        m.weight.data.uniform_(-bound, bound)
        m.bias.data.fill_(0)


def weights_init_identity(m):
    '''
    Initialization function for linear layers, setting weights to the identity
    matrix and bias to zero.
    '''

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.eye_(tensor=m.weight)
        m.bias.data.fill_(0)


def weights_init_rot(m):
    '''
    Initialization function for linear layers, setting weights to a small
    rotation and bias to zero.
    '''

    theta = np.pi / 4
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]])
        m.bias.data.fill_(0)


def main():
    theta = np.pi / 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

    # use gpu if cuda is available
    if torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cpu') # always use cpu because it's faster in this case
    else:
        device = torch.device('cpu')
    print(f'Device used: {device}')

    # create toy data as a tensor
    origin_space, target_space = generate_data(rot_matrix=rot_matrix,
                                               sample_size=10000,
                                               plot=False,
                                               shuffle=True)
    data = np.hstack((origin_space, target_space))
    data_tensor = torch.tensor(data=data,
                               dtype=torch.float,
                               device=device)

    # create neural nets, define optimizers and loss criterion
    map_net = MappingNet().to(device)
    discrim_net = DiscriminatorNet().to(device)
    map_optimizer = torch.optim.Adam(params=map_net.parameters(),
                                     lr=1e-4,
                                     weight_decay=1e-4)
    discrim_optimizer = torch.optim.Adam(params=discrim_net.parameters(),
                                         lr=1e-4,
                                         weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    # Initialize weights to (attempt to) improve stability
    map_net.apply(weights_init_rot)
    discrim_net.apply(weights_init_uniform)

    # begin training loops
    print('Epoch | Iteration | Discrim Loss / 2 | Map Loss')
    print('-----------------------------------------------')

    t0 = time.time()
    for epoch in range(2):

        iteration = 0
        for row in data_tensor:
            iteration += 1

            # get input and target
            origin = row[:2].view(1, 2)
            target = row[2:].view(1, 2)

            # define source tensors
            origin_source = torch.tensor(data=[[1.]]).to(device)
            target_source = torch.tensor(data=[[0.]]).to(device)

            # train discriminator on the true target point
            discrim_loss = training_step(net='discriminator',
                                         discriminator=discrim_net,
                                         optimizer=discrim_optimizer,
                                         loss_criterion=criterion,
                                         point=target,
                                         source=target_source)

            # train each net on a point mapped from the origin space to the target
            mapped_point = map_net(origin)
            map_loss = training_step(net='mapper',
                                     discriminator=discrim_net,
                                     optimizer=map_optimizer,
                                     loss_criterion=criterion,
                                     point=mapped_point,
                                     source=origin_source)
            discrim_loss += training_step(net='discriminator',
                                          discriminator=discrim_net,
                                          optimizer=discrim_optimizer,
                                          loss_criterion=criterion,
                                          point=mapped_point,
                                          source=origin_source)

            if iteration % 500 == 0:
                print(f'{epoch}'.center(6) + '|' +
                      f'{iteration}'.center(11) + '|' +
                      f'{round(discrim_loss/2, 5)}'.center(18) + '|' +
                      f'{round(map_loss, 5)}'.center(9))

    # Print information from the training
    t1 = time.time()
    print(f'Time taken to finish: {t1 - t0}')

    print(f'True Rotation:\n{rot_matrix}')
    print(f'Learned Rotation:\n{map_net.fc1.weight.cpu().detach().numpy()}')
    print(f'Learned Rotation Bias:\n{map_net.fc1.bias.cpu().detach().numpy()}')


    # generate test data and make predictions
    test_origin, test_target = generate_data(rot_matrix=rot_matrix,
                                             sample_size=1000,
                                             plot=False)
    test_tensor = torch.from_numpy(test_origin).float().to(device)
    test_prediction = map_net(test_tensor).cpu().detach().numpy()

    check_pred = test_origin @ map_net.fc1.weight.cpu().detach().numpy() + map_net.fc1.bias.cpu().detach().numpy()

    # plot test data, its transformation, and the true target space
    plt.clf()
    plt.scatter(x=test_origin[:, 0], y=test_origin[:, 1],
                c='red', label='origin space')
    plt.scatter(x=test_target[:, 0], y=test_target[:, 1],
                c='blue', label='target space')
    plt.scatter(x=test_prediction[:, 0], y=test_prediction[:, 1],
                c='yellow', label='transformed data')
    plt.scatter(x=check_pred[:, 0], y=check_pred[:, 1],
                c='green', label='check')
    plt.axis([-5, 5, -5, 5])
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
