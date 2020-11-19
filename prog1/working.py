%matplotlib inline
import numpy as np
from queue import LifoQueue
from random import choice, shuffle, uniform
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import choose


class grid:
    def __init__(self, N, S, F):
        ## Make sure start and end are within the grid
        assert N > 2
        assert S[0] < N
        assert S[1] < N
        assert F[0] < N
        assert F[1] < N

        assert S[0] > 0
        assert S[1] > 0
        assert F[0] > 0
        assert F[1] > 0

        self.N = N

        ## Initialize grid with obstacles
        self.grid = np.ones((N, N), dtype=np.int32)

        ## Start and end position have no obstacles
        self.grid[S] = 0
        # self.grid[F]=0
        obstacle_free_points = {S}

    ### YOUR CODE HERE
    def add_walls(self, coords, walls):
        (y, x) = coords

        if y - 1 > 0 and self.grid[(y - 1, x)] == 1:
            walls.append((y - 1, x))
        if y + 1 < self.N and self.grid[(y + 1, x)] == 1:
            walls.append((y + 1, x))
        if x - 1 > 0 and self.grid[(y, x - 1)] == 1:
            walls.append((y, x - 1))
        if x + 1 < self.N and self.grid[(y, x + 1)] == 1:
            walls.append((y, x + 1))

    def neighbor_x(self, coords, walls):
        (cell_y, cell_x) = coords
        if (
            cell_x - 1 > 0
            and cell_x + 1 < self.N
            and self.grid[(cell_y, cell_x - 1)] == 0
            and self.grid[(cell_y, cell_x + 1)] == 1
        ):
            self.grid[(cell_y, cell_x)] = 0

            self.grid[(cell_y, cell_x + 1)] = 0
            self.add_walls((cell_y, cell_x + 1), walls)
            return True

        elif (
            cell_x - 1 > 0
            and cell_x + 1 < self.N
            and self.grid[(cell_y, cell_x - 1)] == 1
            and self.grid[(cell_y, cell_x + 1)] == 0
        ):
            self.grid[(cell_y, cell_x)] = 0

            self.grid[(cell_y, cell_x - 1)] = 0
            self.add_walls((cell_y, cell_x - 1), walls)
            return True
        return False

    def neighbor_y(self, coords, walls):
        (cell_y, cell_x) = coords
        if (
            cell_y - 1 > 0
            and cell_y + 1 < self.N
            and self.grid[(cell_y - 1, cell_x)] == 0
            and self.grid[(cell_y + 1, cell_x)] == 1
        ):
            self.grid[(cell_y, cell_x)] = 0

            self.grid[(cell_y + 1, cell_x)] = 0
            self.add_walls((cell_y + 1, cell_x), walls)
            return True

        elif (
            cell_y - 1 > 0
            and cell_y + 1 < self.N
            and self.grid[(cell_y - 1, cell_x)] == 1
            and self.grid[(cell_y + 1, cell_x)] == 0
        ):
            self.grid[(cell_y, cell_x)] = 0

            self.grid[(cell_y - 1, cell_x)] = 0
            self.add_walls((cell_y - 1, cell_x), walls)
            return True
        return False

    def solution(self):
        walls = []

        # add starting point walls
        self.add_walls(S, walls)

        while walls:
            shuffle(walls)
            (cell_y, cell_x) = walls.pop()
            if self.grid[(cell_y, cell_x)] == 0:
                continue

            directions = [
                self.neighbor_x((cell_y, cell_x), walls),
                self.neighbor_y((cell_y, cell_x), walls),
            ]
            while directions:
                shuffle(directions)
                dir = directions.pop()

    def draw_map(self, S=None, F=None, path=None):
        image = np.zeros((self.N, self.N, 3), dtype=int)
        image[self.grid == 0] = [255, 255, 255]
        image[self.grid == 1] = [0, 0, 0]
        if S:
            image[S] = [50, 168, 64]
        if F:
            image[F] = [168, 50, 50]
        if path:
            for n in path[1:-1]:
                image[n] = [66, 221, 245]

        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.show()
