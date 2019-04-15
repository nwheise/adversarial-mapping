import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import time


class TransformerNet(torch.nn.Module):
    
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=2, out_features=2)
    
    def forward(self, x):
        x = self.fc1(x)
        return x


def points_on_triangle(vertices: np.ndarray, n: int) -> np.ndarray:
    '''
    Give n random points uniformly on a triangle.

    Parameters:
        vertices: numpy array of shape (2, 3), giving triangle vertices as one
                  vertex per row
        n: number of points to generate
    '''
    x = np.sort(np.random.rand(2, n), axis=0)
    return np.column_stack([x[0], x[1] - x[0], 1 - x[1]]) @ vertices


def generate_data(sample_size=10000, plot=False):
    '''
    Generate some toy data. The origin space lies in a triangle, and the
    target space lies in a triangle produced by rotating the first by an angle.

    Parameters:
        sample_size: number of samples to generate
        plot: boolean to display plotted data or not
    '''

    # Generate data in a triangle
    triangle_vertices = np.array([(1, 1), (3, 5), (2, 6)])
    points = points_on_triangle(triangle_vertices, sample_size)

    # Rotation by theta
    theta = np.pi / 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                           [np.sin(theta), np.cos(theta)]])
    points_rotated = points @ rot_matrix

    # Optionally display the plot
    if plot:
        plt.scatter(x=points[:, 0], y=points[:, 1], c='red')
        plt.scatter(x=points_rotated[:, 0], y=points_rotated[:, 1], c='blue')
        plt.show()

    return points, points_rotated


def main():
    # create toy data as a tensor
    origin_space, target_space = generate_data(sample_size=25000, plot=True)
    data = np.hstack((origin_space, target_space))
    data_tensor = torch.from_numpy(data).float()
    # data_tensor = data_tensor.cuda()

    # create neural net, define optimizer and loss criterion
    net = TransformerNet()
    # net = net.cuda()
    optimizer = torch.optim.Adam(params=net.parameters())
    criterion = torch.nn.MSELoss()

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
    test_tensor = torch.from_numpy(test_origin).float()
    test_prediction = net(test_tensor).detach().numpy()

    # plot test data and its transformation
    plt.clf()
    plt.scatter(x=test_origin[:, 0], y=test_origin[:, 1], c='red')
    plt.scatter(x=test_prediction[:, 0], y=test_prediction[:, 1], c='blue')
    plt.show()


if __name__ == '__main__':
    main()