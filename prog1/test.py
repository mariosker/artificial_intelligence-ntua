# import math
from queue import LifoQueue
from random import shuffle, uniform
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from numpy.core.fromnumeric import choose


class visualization:

    def __init__(self, S, F):
        '''
          Η μέθοδος αυτή αρχικοποιεί ένα αντικείμενο τύπου visualization.
          Είσοδος:
          -> S: το σημείο εκκίνσης της αναζήτησης
          -> F: το σημείο τερματισμού
        '''
        self.S = S
        self.F = F
        self.images = []

    def draw_step(self, grid, frontier, expanded_nodes):
        '''
          Η συνάρτηση αυτή καλείται για να σχεδιαστεί ένα frame στο animation (πρακτικά έπειτα από την επέκταση κάθε κόμβου)
          Είσοδος:
          -> grid: Ένα χάρτης τύπου grid
          -> frontier: Μια λίστα με τους κόμβους που ανήκουν στο μέτωπο της αναζήτησης
          -> expanded_nodes: Μια λίστα με τους κόμβους που έχουν ήδη επεκταθεί
          Επιστρέφει: None
          Η συνάρτηση αυτή πρέπει να καλεστεί τουλάχιστον μια φορά για να μπορέσει να σχεδιαστει ένα animation (πρεπεί το animation να έχει τουλάχιστον ένα frame).
        '''
        image = np.zeros((grid.N, grid.N, 3), dtype=int)
        image[grid.grid == 0] = [255, 255, 255]
        image[grid.grid == 1] = [0, 0, 0]

        for node in expanded_nodes:
            image[node] = [0, 0, 128]

        for node in frontier:
            image[node] = [0, 225, 0]

        image[self.S] = [50, 168, 64]
        image[self.F] = [168, 50, 50]
        self.images.append(image)

    def add_path(self, path):
        '''
          Η συνάρτηση αυτή προσθέτει στο τελευταίο frame το βέλτιστο μονοπάτι.
          Είσοδος:
          -> path: Μια λίστα η όποια περιέχει το βέλτιστο μονοπάτι (η οποία πρέπει να περιέχει και τον κόμβο αρχή και τον κόμβο στόχο)
          Έξοδος: None
        '''
        for n in path[1:-1]:
            image = np.copy(self.images[-1])
            image[n] = [66, 221, 245]
            self.images.append(image)
        for _ in range(100):
            self.images.append(image)

    def create_gif(self, fps=30, repeat_delay=2000):
        if len(self.images) == 0:
            raise EmptyStackOfImages("Error! You have to call 'draw_step' at  first.")
        fig = plt.figure()
        plt.axis('off')
        ims = []
        for img in self.images:
            img = plt.imshow(img)
            ims.append([img])
        ani = animation.ArtistAnimation(fig, ims, interval=1000 // fps, blit=True, repeat_delay=repeat_delay)
        plt.close(fig)
        return ani

    def save_gif(self, filename, fps=30):
        '''
            Η συνάρτηση αυτή ξαναδημιουργεί και αποθηκεύει το animation σε ένα αρχείο.
            Είσοδος:
            -> Το όνομα του αρχείου με κατάληξη .gif
            Έξοδος: (None)
        '''
        ani = self.create_gif(fps)
        writer = PillowWriter(fps=fps)
        ani.save(filename, writer=writer)

    def show_gif(self, fps=30, repeat_delay=2000):
        '''
            Η συνάρτηση αυτή εμφανίζει inline το animation.
            Είσοδος:
            -> fps: τα frames per second
            Έξοδος: Το αντικείμενο που παίζει το animation
            Exceptions: EmptyStackOfImages αν το animation δεν έχει ούτε ένα frame, δηλαδή αν η draw_step δεν έχει καλεστεί ποτέ.
        '''
        ani = self.create_gif(fps, repeat_delay)
        # return HTML(ani.to_html5_video())
        return HTML(ani.to_jshtml())

    def show_last_frame(self):
        '''
            Η μέθοδος αυτή εμφανίζει inline το τελευταίο frame που έχει δημιουργήθει.
            Είσοδος:
            Έξοδος: Το αντικείμενο που εμφανίζει την εικόνα.
            Exceptions: EmptyStackOfImages αν το animation δεν έχει ούτε ένα frame, δηλαδή αν η draw_step δεν έχει καλεστεί ποτέ.
        '''
        if len(self.images) == 0:
            raise EmptyStackOfImages("Error! You have to call 'draw_step' at  first.")
        else:
            plt.imshow(self.images[-1])


class EmptyStackOfImages(Exception):
    pass


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
        self.grid[F] = 0
        obstacle_free_points = {S, F}

        ### YOUR CODE HERE
        def add_walls(coords, walls):
            (y, x) = coords

            if y - 1 > 0 and self.grid[(y - 1, x)] == 1:
                walls.append((y - 1, x))
            if y + 1 < self.N and self.grid[(y + 1, x)] == 1:
                walls.append((y + 1, x))
            if x - 1 > 0 and self.grid[(y, x - 1)] == 1:
                walls.append((y, x - 1))
            if x + 1 < self.N and self.grid[(y, x + 1)] == 1:
                walls.append((y, x + 1))

        def is_cell_obstructed(coords):
            (y, x) = coords
            if y - 1 > 0 and self.grid[(y - 1, x)] == 0:
                return False
            if y + 1 < self.N and self.grid[(y + 1, x)] == 0:
                return False
            if x - 1 > 0 and self.grid[(y, x - 1)] == 0:
                return False
            if x + 1 < self.N and self.grid[(y, x + 1)] == 0:
                return False
            return True

        def neighbors_in_x_axis(coords, walls):
            (cell_y, cell_x) = coords
            if (cell_x - 1 > 0 and cell_x + 1 < self.N and self.grid[(cell_y, cell_x - 1)] == 0 and
                    self.grid[(cell_y, cell_x + 1)] == 1):
                self.grid[(cell_y, cell_x)] = 0

                self.grid[(cell_y, cell_x + 1)] = 0
                add_walls((cell_y, cell_x + 1), walls)
                return True

            elif (cell_x - 1 > 0 and cell_x + 1 < self.N and self.grid[(cell_y, cell_x - 1)] == 1 and
                  self.grid[(cell_y, cell_x + 1)] == 0):
                self.grid[(cell_y, cell_x)] = 0

                self.grid[(cell_y, cell_x - 1)] = 0
                add_walls((cell_y, cell_x - 1), walls)
                return True

            elif (cell_x - 1 > 0 and cell_x + 1 < self.N and (cell_y, cell_x - 1) in obstacle_free_points and
                  is_cell_obstructed((cell_y, cell_x - 1))):
                self.grid[(cell_y, cell_x)] = 0
                return True

            elif (cell_x - 1 > 0 and cell_x + 1 < self.N and (cell_y, cell_x + 1) in obstacle_free_points and
                  is_cell_obstructed((cell_y, cell_x + 1))):
                self.grid[(cell_y, cell_x)] = 0
                return True

            return False

        def neighbors_in_y_axis(coords, walls):
            (cell_y, cell_x) = coords
            if (cell_y - 1 > 0 and cell_y + 1 < self.N and self.grid[(cell_y - 1, cell_x)] == 0 and
                    self.grid[(cell_y + 1, cell_x)] == 1):
                self.grid[(cell_y, cell_x)] = 0

                self.grid[(cell_y + 1, cell_x)] = 0
                add_walls((cell_y + 1, cell_x), walls)
                return True

            elif (cell_y - 1 > 0 and cell_y + 1 < self.N and self.grid[(cell_y - 1, cell_x)] == 1 and
                  self.grid[(cell_y + 1, cell_x)] == 0):
                self.grid[(cell_y, cell_x)] = 0

                self.grid[(cell_y - 1, cell_x)] = 0
                add_walls((cell_y - 1, cell_x), walls)
                return True

            elif (cell_y - 1 > 0 and cell_y + 1 < self.N and (cell_y - 1, cell_x) in obstacle_free_points and
                  is_cell_obstructed((cell_y - 1, cell_x))):
                self.grid[(cell_y, cell_x)] = 0
                return True

            elif (cell_y - 1 > 0 and cell_y + 1 < self.N and (cell_y + 1, cell_x) in obstacle_free_points and
                  is_cell_obstructed((cell_y + 1, cell_x))):
                self.grid[(cell_y, cell_x)] = 0
                return True

            return False

        def generate_maze():
            walls = []

            add_walls(S, walls)

            while walls:
                shuffle(walls)
                (cell_y, cell_x) = walls.pop()

                if self.grid[(cell_y, cell_x)] == 0:
                    continue

                neighbors_in_x_axis((cell_y, cell_x), walls)
                neighbors_in_y_axis((cell_y, cell_x), walls)

        generate_maze()

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

    def manhattan(self, start_coords, end_coords):
        (sy, sx) = start_coords
        (fy, fx) = end_coords
        return (abs(sx - fx) + abs(sy - fy)) * self.heuristic(sy, fy)

    def euclidean(self, start_coords, end_coords):
        (sy, sx) = start_coords
        (fy, fx) = end_coords
        return np.sqrt(((sx - fx) * self.heuristic(sx, fx))**2 + ((sy - fy) * self.heuristic(sy, fy))**2)

    def find_path(self, heuristic_fun):

        def get_neighbors(coords):
            neighbors = []
            (y, x) = coords
            if (x - 1 > 0 and self.grid.grid[(y, x - 1)] != 1):
                neighbors.append((y, x - 1))
            if (x + 1 < self.grid.N and self.grid.grid[(y, x + 1)] != 1):
                neighbors.append((y, x + 1))
            if (y - 1 > 0 and self.grid.grid[(y - 1, x)] != 1):
                neighbors.append((y - 1, x))
            if (y + 1 < self.grid.N and self.grid.grid[(y + 1, x)] != 1):
                neighbors.append((y + 1, x))
            return neighbors

        g_score = {self.S: 0}
        f_score = {self.S: heuristic_fun(self.S, self.F)}
        frontier = [(f_score[self.S], self.S)]
        expanded_nodes = set()
        parent = {}

        while frontier:
            frontier.sort(key=lambda x: x[0], reverse=True)
            current = frontier.pop()[1]

            if current == self.F:
                while current in parent:
                    self.path.append(current)
                    current = parent[current]
                self.path.append(current)
                return True

            expanded_nodes.add(current)

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + heuristic_fun(current, neighbor)

                if neighbor in expanded_nodes and tentative_g_score >= g_score.get(neighbor, 0):
                    continue
                if tentative_g_score < g_score.get(neighbor, 0) or neighbor not in [i[1] for i in frontier]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic_fun(neighbor, self.F)
                    frontier.append((f_score[neighbor], neighbor))
            pf.vis.draw_step(map, [i[1] for i in frontier], expanded_nodes)
        return False

    def get_path(self):
        return self.path


N = 100
S = (5, 9)
F = (95, 85)

map = grid(N, S, F)
pf = pathfinder(S, F, map, lambda x, y: 1, lambda x, y: 0)
map.draw_map(S, F, pf.get_path())
pf.vis.add_path(pf.get_path())

#καλούμε την μέθοδο για να παρουσιάσουμε το animation στο Notebook
# pf.vis.save_gif("mygif.gif")
pf.vis.show_gif(fps = 1)