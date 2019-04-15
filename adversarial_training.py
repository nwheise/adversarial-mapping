import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from nets import MappingNet
from nets import DiscriminatorNet
from toy_data import generate_data


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
    origin_space, target_space = generate_data(plot=True)
    data = np.hstack((origin_space, target_space))
    data_tensor = torch.from_numpy(data).float().to(device)

    # create neural net, define optimizer and loss criterion
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
    iteration = 0
    loss_curve = []
    t0 = time.time()
    for i in data_tensor:
        iteration += 1

        # get input and target
        origin = i[:2]
        target = i[2:]

        # zero the parameter gradients
        map_optimizer.zero_grad()

        # forward pass
        outputs = map_net(origin)

        # adversarial
        discrim_data_tensor = torch.cat((target.view(1, 2),
                                         outputs.view(1, 2)), dim=0) \
                                   .to(device)
        discrim_output = discrim_net(discrim_data_tensor).to(device)
        classes = torch.Tensor(data=[[1, 0], [0, 1]]).to(device)
        disc = torch.cat(tensors=(discrim_output, classes), dim=1).to(device)

        # calculate loss by trying to predict if an input came from original
        # space after transformation or from the target space
        for j in disc:
            discrim_optimizer.zero_grad()
            loss = criterion(j[:2], j[2:].detach())
            loss.backward(retain_graph=True)
            discrim_optimizer.step()

        # optimize
        map_optimizer.step()

        # print loss every 1000 samples
        if iteration % 1000 == 0:
            print(f'Loss: {loss:.10f} after {iteration} samples')

        # add loss to a list to be plotted
        loss_curve.append(loss.data.item())

    t1 = time.time()
    print(f'Time taken to finish: {t1 - t0}')

    # plot loss curve
    plt.clf()
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.show()

    # generate test data and make predictions
    test_origin, test_target = generate_data(sample_size=10000, plot=False)
    test_tensor = torch.from_numpy(test_origin).float().to(device)
    test_prediction = map_net(test_tensor).cpu().detach().numpy()

    # plot test data and its transformation
    plt.clf()
    plt.scatter(x=test_origin[:, 0], y=test_origin[:, 1], c='red')
    plt.scatter(x=test_prediction[:, 0], y=test_prediction[:, 1], c='blue')
    plt.show()


if __name__ == '__main__':
    main()