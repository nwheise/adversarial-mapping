import numpy as np
import matplotlib.pyplot as plt
from nets import MappingNet


def points_on_triangle(vertices: np.ndarray, n: int) -> np.ndarray:
    '''
    Give n random points uniformly on a triangle.

    Parameters:
        vertices: numpy array of shape (3, 2), giving triangle vertices as one
                  vertex per row
        n: number of points to generate
    '''
    x = np.sort(np.random.rand(2, n), axis=0)
    x = np.column_stack([x[0], x[1] - x[0], 1 - x[1]]) @ vertices
    np.random.shuffle(x)

    return x


def generate_data(sample_size=25000, plot=False):
    '''
    Generate some toy data. The origin space lies in a triangle, and the
    target space lies in a triangle produced by rotating the first by an angle.

    Parameters:
        sample_size: number of samples to generate
        plot: boolean to display plotted data or not
    '''

    # Generate data in a triangle
    triangle_vertices = np.array([(1, 1), (3, 4), (1, 3)])
    points = points_on_triangle(triangle_vertices, sample_size)

    # Rotation by theta
    theta = np.pi / 2
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                           [np.sin(theta), np.cos(theta)]])
    points_rotated = points @ rot_matrix

    # Optionally display the plot
    if plot:
        plt.scatter(x=points[:, 0], y=points[:, 1],
                    c='red', label='origin space')
        plt.scatter(x=points_rotated[:, 0], y=points_rotated[:, 1],
                    c='blue', label='target space')
        plt.axis([-5, 5, -5, 5])
        plt.legend(loc='upper left')
        plt.show()

    return points, points_rotated
