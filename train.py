#!/usr/bin/env python3

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch

from nets import MappingNet, DiscriminatorNet
from toy_data import generate_data

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

writer = SummaryWriter()


def training_step(
    net: str,
    discriminator: DiscriminatorNet,
    optimizer: torch.optim.Optimizer,
    loss_criterion: torch.nn.modules.loss._Loss,
    point: torch.Tensor,
    source: torch.Tensor,
):
    """
    Performs the training step on either the mapper or the discriminator.

    :param net: 'mapper' or 'discriminator'
    :param discriminator: discriminator neural network
    :param optimizer: optimizer for the neural network being trained
    :param loss_criterion: loss function
    :param point: data vector of shape (1, 2)
    :param source: vector of shape (1, 2) describing the source of
        the point, either a mapped point or a true target point
    """

    # zero grad the optimizer
    optimizer.zero_grad()

    # get discriminator output for the given point
    discrim_output = discriminator(point)

    # calculate loss, the mapper tries to make the output incorrect and the
    # discriminator tries to make the output match the source
    if net == "mapper":
        loss = loss_criterion(discrim_output, 1 - source)
    elif net == "discriminator":
        loss = loss_criterion(discrim_output, source)
    else:
        raise Exception(f'Invalid parameter for "net": {net}')

    # backward propagate
    loss.backward(retain_graph=True)

    # optimizer step
    optimizer.step()

    return loss.data.item()


def main():
    # Use CPU for this simple model
    device = torch.device("cpu")

    # Generate a toy training dataset
    points, points_rotated = generate_data(sample_size=10000)
    data = np.hstack((points, points_rotated))
    data_tensor = torch.tensor(data=data, dtype=torch.float, device=device)

    # Generate a toy test dataset
    test_origin, test_target = generate_data(sample_size=1000)
    test_tensor = torch.tensor(
        data=test_origin, dtype=torch.float32, device=device
    )

    # create neural nets, define optimizers and loss criterion
    map_net = MappingNet().to(device)
    discrim_net = DiscriminatorNet().to(device)
    map_optimizer = torch.optim.Adam(params=map_net.parameters(), lr=1e-4)
    discrim_optimizer = torch.optim.Adam(
        params=discrim_net.parameters(), lr=1e-4
    )
    criterion = torch.nn.BCELoss()

    # begin training loops
    logger.info("Training has begun!")
    logger.info("Access TensorBoard to view results.")

    t0 = time.time()
    global_step = 0
    for epoch in range(25):

        for data_pair in data_tensor:
            global_step += 1

            # get input and target
            origin = data_pair[:2].view(1, 2)
            target = data_pair[2:].view(1, 2)

            # define source tensors
            origin_source = torch.tensor(data=[[1.0]], device=device)
            target_source = torch.tensor(data=[[0.0]], device=device)

            # train discriminator on the true target point
            discrim_loss = training_step(
                net="discriminator",
                discriminator=discrim_net,
                optimizer=discrim_optimizer,
                loss_criterion=criterion,
                point=target,
                source=target_source,
            )

            # train each net on a point mapped from the origin to the target
            mapped_point = map_net(origin)
            map_loss = training_step(
                net="mapper",
                discriminator=discrim_net,
                optimizer=map_optimizer,
                loss_criterion=criterion,
                point=mapped_point,
                source=origin_source,
            )
            discrim_loss = training_step(
                net="discriminator",
                discriminator=discrim_net,
                optimizer=discrim_optimizer,
                loss_criterion=criterion,
                point=mapped_point,
                source=origin_source,
            )

            # Logging in tensorboard
            writer.add_scalar(
                tag="loss/mapper",
                scalar_value=map_loss,
                global_step=global_step,
            )
            writer.add_scalar(
                tag="loss/discriminator",
                scalar_value=discrim_loss,
                global_step=global_step,
            )

            if not global_step % 500:
                logger.info(f"Global Step : {global_step}")

        # predict on test data
        test_pred = map_net(test_tensor).data.numpy()

        # plot test data, its transformation, and the true target space
        plt.clf()
        plt.scatter(
            x=test_origin[:, 0],
            y=test_origin[:, 1],
            c="red",
            label="origin space",
        )
        plt.scatter(
            x=test_target[:, 0],
            y=test_target[:, 1],
            c="blue",
            label="target space",
        )
        plt.scatter(
            x=test_pred[:, 0],
            y=test_pred[:, 1],
            c="yellow",
            label="transformed data",
        )
        plt.axis([-5, 5, -5, 5])
        plt.legend(loc="upper left")

        fig = plt.gcf()

        writer.add_figure(tag="plot", figure=fig, global_step=global_step)

    # Print information from the training
    t1 = time.time()
    logger.info(f"Time taken to finish: {t1 - t0}")


if __name__ == "__main__":
    main()
