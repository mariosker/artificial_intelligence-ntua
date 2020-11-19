# from visualization import *
import math

class pathfinder:
    def __init__(self, S, F, grid, c, h):
        self.S = S
        self.F = F
        self.grid = grid
        self.cost = c
        self.heuristic = h
        #for visualization
        self.vis = visualization(S, F)
        self.path = []

    def manhattan(self, coords):
        (Ys, Xs) = coords
        (Yf, Xf) = self.F
        return (abs(Xs - Xf) + abs(Ys - Yf))

    def euclidean(self, coords):
        (y, x) = coords
        (fy, fx) = self.F
        return math.sqrt(((x - fx))**2 + ((y - fy))**2)

    def find_frontier(self, coords):
        frontier = []
        (y,x) = coords
        if(x-1 > 0 and self.grid((y,x-1)) != 1):
            frontier.append((y,x-1))
        if(x+1 < self.N and self.grid((y,x+1)) != 1):
            frontier.append((y,x+1))
        if(y-1 > 0 and self.grid((y-1,x)) != 1):
            frontier.append((y-1,x))
        if(y+1 < self.N and self.grid((y+1,x)) != 1):
            frontier.append((y+1,x))
        return frontier

    def find_path(self, heuristic):
        (y, x) = self.S
        frontier = [self.S]
        expanded_nodes = []

        while frontier:
            frontier.extend(find_frontier(coords))


    def get_path(self):
        return self.path
