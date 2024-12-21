import mpi4py.MPI as MPI
import sys

class Unit:
    def __init__(self, type, health, attack, heal_rate, attack_directions):
        self.type = type
        self.health = health
        self.attack = attack
        self.heal_rate = heal_rate
        self.attack_directions = []

class Earth(Unit):
    def __init__(self):
        self.attack_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        super().__init__("E", 18, 2, 3, self.attack_directions)

class Fire(Unit):
    def __init__(self):
        self.attack_directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        super().__init__("F", 12, 4, 1, self.attack_directions)

class Water(Unit):
    def __init__(self):
        self.attack_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        super().__init__("W", 14, 3, 2, self.attack_directions)

class Air(Unit):
    def __init__(self):
        self.attack_directions = [(1, 0), (0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
        super().__init__("A", 10, 2, 2, self.attack_directions)

class Pixel:
    def __init__(self, unit):
        self.unit = unit
        self.damage_to_be_taken = 0

    def __str__(self):
        return self.unit.type

    def copy(self):
        return Pixel(self.unit)

# processes will have their own grid, and they will
class Grid:
    def __init__(self, n):
        self.n = n
        self.grid = [[None for _ in range(n)] for _ in range(n)]
        self.num_air = 0
        self.num_fire = 0
        self.num_water = 0
        self.num_earth = 0

        # map from (row, column) to unit pointer
        self.air_units = {}
        self.fire_units = {}
        self.water_units = {}
        self.earth_units = {}


    def __str__(self):
        ret = " "
        for i in range(self.n):
            for j in range(self.n):
                if self.grid[i][j] is not None:
                    ret += self.grid[i][j].type + " "
                else:
                    ret += ". "
            ret += "\n"
        return ret

    def get(self, i, j):
        if 0 <= i < self.n and 0 <= j < self.n:
            return self.grid[i][j]
        else:
            return -1

    def set(self, i, j, pixel):
        self.grid[i][j] = pixel
        if pixel is not None and pixel.unit is not None:
            if pixel.unit.type == "A":
                self.num_air += 1
                self.air_units[(i, j)] = pixel.unit
            elif pixel.unit.type == "F":
                self.num_fire += 1
                self.fire_units[(i, j)] = pixel.unit
            elif pixel.unit.type == "W":
                self.num_water += 1
                self.water_units[(i, j)] = pixel.unit
            elif pixel.unit.type == "E":
                self.num_earth += 1
                self.earth_units[(i, j)] = pixel.unit

        else: # then (i, j) is being deleted
            if (i, j) in self.air_units:
                self.num_air -= 1
                del self.air_units[(i, j)]
            elif (i, j) in self.fire_units:
                self.num_fire -= 1
                del self.fire_units[(i, j)]
            elif (i, j) in self.water_units:
                self.num_water -= 1
                del self.water_units[(i, j)]
            elif (i, j) in self.earth_units:
                self.num_earth -= 1
                del self.earth_units[(i, j)]




def create_unit(unit_type):
    if unit_type == "E":
        return Earth()
    elif unit_type == "F":
        return Fire()
    elif unit_type == "W":
        return Water()
    elif unit_type == "A":
        return Air()
    else:
        return None

def get_worker_borders(worker_rank, num_workers, n):
    """
    :param n: the number of lines each worker will be responsible
    :return: (left, right, top, bottom) boundary indexes of the area that the worker will be responsible for. right and
    bottom boundaries are exclusive, left and top boundaries are inclusive.
    """

    worker_rank -= 1 # so that x and y values start from 0

    num_workers_per_row = int(num_workers ** 0.5)

    worker_x = worker_rank % num_workers_per_row
    worker_y = worker_rank // num_workers_per_row

    left_boundary = worker_x * n
    right_boundary = (worker_x + 1) * n  # exclusive
    top_boundary = worker_y * n
    bottom_boundary = (worker_y + 1) * n  # exclusive

    return left_boundary, right_boundary, top_boundary, bottom_boundary

def send_to_horizontal_neighbors(data, rank, num_workers, comm):
    """
    :param grid: the grid that will be sent to neighbors
    :param rank: the rank of the process
    :param num_workers: the total number of workers
    :param comm: the communicator object
    :return: None
    """

    num_workers_per_row = int(num_workers ** 0.5)
    rank = rank - 1  # only the worker ranks are considered

    if not rank % num_workers_per_row == 0:  # check if leftmost process
        comm.send(data[0], dest=rank - 1)

    if not rank % num_workers_per_row == num_workers_per_row - 1:  # check if rightmost process
        comm.send(data[1], dest=rank + 1)
    

def send_to_vertical_neighbors(data, rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    rank = rank - 1  # only the worker ranks are considered

    if not rank < num_workers_per_row:
        comm.send(data[0], dest=rank - num_workers_per_row)
    if not rank >= num_workers - num_workers_per_row:
        comm.send(data[1], dest=rank + num_workers_per_row)
    

def send_to_cross_neighbors(data, rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    rank = rank - 1  # only the worker ranks are considered

    if not rank % num_workers_per_row == 0 and not rank < num_workers_per_row:
        comm.send(data[0], dest=rank - num_workers_per_row - 1)
    if not rank % num_workers_per_row == num_workers_per_row - 1 and not rank < num_workers_per_row:
        comm.send(data[1], dest=rank - num_workers_per_row + 1)
    if not rank % num_workers_per_row == num_workers_per_row - 1 and not rank >= num_workers - num_workers_per_row:
        comm.send(data[2], dest=rank + num_workers_per_row + 1)
    if not rank % num_workers_per_row == 0 and not rank >= num_workers - num_workers_per_row:
        comm.send(data[3], dest=rank + num_workers_per_row - 1)
    
    


def receive_from_horizontal_neighbors(rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    rank = rank - 1  # only the worker ranks are considered

    left_data = None
    right_data = None

    if not rank % num_workers_per_row == num_workers_per_row - 1:  # check if rightmost process
        right_data = comm.recv(source=rank + 1)
    if not rank % num_workers_per_row == 0:  # check if leftmost process
        left_data = comm.recv(source=rank - 1)

    return left_data, right_data

def receive_from_vertical_neighbors(rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    rank = rank - 1  # only the worker ranks are considered

    top_data = None
    bottom_data = None

    if not rank >= num_workers - num_workers_per_row:
        bottom_data = comm.recv(source=rank + num_workers_per_row)
    if not rank < num_workers_per_row:
        top_data = comm.recv(source=rank - num_workers_per_row)

    return top_data, bottom_data

def receive_from_cross_neighbors(rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    rank = rank - 1  # only the worker ranks are considered

    top_left_data = None
    top_right_data = None
    bottom_left_data = None
    bottom_right_data = None

    if not rank % num_workers_per_row == num_workers_per_row - 1 and not rank >= num_workers - num_workers_per_row:
        bottom_right_data = comm.recv(source=rank + num_workers_per_row + 1)
    if not rank % num_workers_per_row == 0 and not rank < num_workers_per_row:
        top_left_data = comm.recv(source=rank - num_workers_per_row - 1)
    if not rank % num_workers_per_row == num_workers_per_row - 1 and not rank < num_workers_per_row:
        top_right_data = comm.recv(source=rank - num_workers_per_row + 1)
    if not rank % num_workers_per_row == 0 and not rank >= num_workers - num_workers_per_row:
        bottom_left_data = comm.recv(source=rank + num_workers_per_row - 1)

    return top_left_data, top_right_data, bottom_left_data, bottom_right_data


def point_letter(row, column):
    l = ["AB", "CD"]
    return l[row%2][column%2]
    

def communicate(data, rank, num_workers, comm):
    """
    :param data: the grid that will be sent to neighbors, ordered as [upperleft, up, upperright, left, right, lowerleft, down, lowerright]
    :param rank: the rank of the process
    :param num_workers: the total number of workers
    :param comm: the communicator object
    :return: the received data from neighbors, ordered as [upperleft, up, upperright, left, right, lowerleft, down, lowerright]
    """

    num_workers_per_row = int(num_workers ** 0.5)
    process_row = rank // num_workers_per_row
    process_col = rank % num_workers_per_row
    
    

    if point_letter(process_row, process_col) == "A":
        send_to_cross_neighbors(data[0,2,5,7], rank, num_workers, comm)
        send_to_horizontal_neighbors(data[3,4], rank, num_workers, comm)
        send_to_vertical_neighbors(data[1,6], rank, num_workers, comm)

        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)
        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)

    elif point_letter(process_row, process_col) == "B":
        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)

        send_to_cross_neighbors(data[0,2,5,7], rank, num_workers, comm)
        send_to_horizontal_neighbors(data[3,4], rank, num_workers, comm)
        send_to_vertical_neighbors(data[1,6], rank, num_workers, comm)

        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)

    elif point_letter(process_row, process_col) == "C":
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)
        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)

        send_to_cross_neighbors(data[0,2,5,7], rank, num_workers, comm)
        send_to_horizontal_neighbors(data[3,4], rank, num_workers, comm)
        send_to_vertical_neighbors(data[1,6], rank, num_workers, comm)

        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)

    elif point_letter(process_row, process_col) == "D":
        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)
        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)

        send_to_cross_neighbors(data[0,2,5,7], rank, num_workers, comm)
        send_to_horizontal_neighbors(data[3,4], rank, num_workers, comm)
        send_to_vertical_neighbors(data[1,6], rank, num_workers, comm)

    return cross_neighbor_data[0], vertical_neighbor_data[0], cross_neighbor_data[1], horizontal_neighbor_data[0], horizontal_neighbor_data[1], cross_neighbor_data[2], vertical_neighbor_data[1], cross_neighbor_data[3] 

def merge_grids(own_grid, data, rank, num_workers, n):
    """
    :param data: the received data from neighbors, ordered as [upperleft, up, upperright, left, right, lowerleft, down, lowerright]
    :param rank: the rank of the process
    :param num_workers: the total number of workers
    :param n: the number of lines each worker will be responsible
    :return: the merged grid
    """

    merged_grid = Grid(3*n)

    for i in range(n):
        for j in range(n):
            merged_grid.set(i, j, data[0].get(i, j))
        for j in range(n, 2*n):
            merged_grid.set(i, j, data[1].get(i, j - n))
        for j in range(2*n, 3*n):
            merged_grid.set(i, j, data[2].get(i, j - 2*n))

    for i in range(n, 2*n):
        for j in range(n):
            merged_grid.set(i, j, data[3].get(i - n, j))
        for j in range(n, 2*n):
            merged_grid.set(i, j, own_grid.get(i - n, j - n))
        for j in range(2*n, 3*n):
            merged_grid.set(i, j, data[4].get(i - n, j - 2*n))

    for i in range(2*n, 3*n):
        for j in range(n):
            merged_grid.set(i, j, data[5].get(i - 2*n, j))
        for j in range(n, 2*n):
            merged_grid.set(i, j, data[6].get(i - 2*n, j - n))
        for j in range(2*n, 3*n):
            merged_grid.set(i, j, data[7].get(i - 2*n, j - 2*n))
    


    return merged_grid

def handle_air_movement(grid, air_unit_coords, n):
    """
    Handles the movement phase for Air units
    Returns a list of movement decisions to be applied simultaneously
    """
    movement_decisions = []  # List of (from_coord, to_coord) tuples
    
    for coord in air_unit_coords:
        best_position = coord
        max_attackable = count_attackable_enemies(grid, coord, n)
        
        pixel = grid.get(coord[0], coord[1])
        # Check all possible movement directions
        for dx, dy in pixel.unit.attack_directions:
            new_x = coord[0] + dx
            new_y = coord[1] + dy
            
            if 0 <= new_x < n and 0 <= new_y < n and grid.get(new_x, new_y) is None:
                attackable = count_attackable_enemies(grid, (new_x, new_y), n)
                
                if attackable > max_attackable:
                    max_attackable = attackable
                    best_position = (new_x, new_y)
                elif attackable == max_attackable and attackable > count_attackable_enemies(grid, coord, n):
                    if new_x < best_position[0] or (new_x == best_position[0] and new_y < best_position[1]):
                        best_position = (new_x, new_y)
        
        if best_position != coord:
            movement_decisions.append((coord, best_position))
    
    return movement_decisions

def count_attackable_enemies(grid, coord, n):
    """
    Counts how many enemy units an Air unit can attack from the given position
    """
    count = 0
    pixel = grid.get(coord[0], coord[1])
    if pixel is None or pixel.unit.type != 'A':
        return 0
        
    # Check all directions including diagonals
    for drow, dcol in pixel.unit.attack_directions:
        row = coord[0] + drow
        col = coord[1] + dcol
        
        if 0 <= row < n and 0 <= col < n:
            if grid.get(row, col) is not None and grid.get(row, col).unit.type != 'A':
                count += 1
            elif grid.get(row, col) is None:
                row += drow
                col += dcol
                if 0 <= row < n and 0 <= col < n and grid.get(row, col) is not None and grid.get(row, col).unit.type != 'A':
                    count += 1
                    
    return count


if __name__ == "__main__":

    # check if the terminal input is correct
    if len(sys.argv) != 3:
        print("Usage: mpiexec -n [P] main.py <input.txt> <output.txt>")
        sys.exit(1)


    # read the input before setting multiple processes
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    input_file = open(input_file_name, "r")
    output_file = open(output_file_name, "w")

    lines = input_file.readlines()
    first_line = lines[0].strip().split(" ")
    line_index = 1

    # IMPORTANT: num_processors being a perfect square that is a divisor of N**2 is assumed
    N = int(first_line[0])
    num_waves = int(first_line[1])
    num_units_per_faction_per_wave = int(first_line[2])
    num_rounds_per_wave = int(first_line[3])


    # set up the MPI
    MPI.Init()
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # will be different for every process, 0 is the manager
    num_workers = comm.Get_size() - 1
    num_workers_per_row = int(num_workers ** 0.5)

    n = N // num_workers  # the number of lines each worker will be responsible

    if rank == 0: # manager

        main_grid = Grid(N)

        for wave in range(num_waves):

            # 1) parse the new wave and create the new units

            while not lines[line_index].startswith("Wave"):
                line_index += 1

            for line_idx in range(1, 5):
                line = lines[line_index + line_idx]
                unit_type = line[0]
                coordinates = line[2:].strip().split(",")

                for coord in coordinates:
                    i, j = coord.split(" ") # x and y are swapped in the input file
                    i = int(i)
                    j = int(j)
                    if main_grid.get(i, j) is None:
                        main_grid.set(i, j, Pixel(create_unit(unit_type)))
                    else:
                        pass # Don't override existing units

            line_index += 5

            # 2) distribute each wave to workers

            for worker_rank in range(1, num_workers + 1):

                # these worker x, y values are the x, y values of processors in the processor-grid
                # the indexes start from 0, that's why I subtracted 1 from the worker_rank
                left, right, top, bottom = get_worker_borders(worker_rank, num_workers, n)

                worker_grid = Grid(n)

                for col in range(left, right):
                    for row in range(top, bottom):
                        if worker_grid.get(row, col) is not None:
                            worker_grid.set(row - top, col - left, main_grid.get(row, col).copy())

                comm.send(worker_grid, dest=worker_rank)


            # 3) collect the results from workers before the next wave

            for worker_rank in range(1, num_workers + 1):
                worker_grid = comm.recv(source=worker_rank)
                left, right, top, bottom = get_worker_borders(worker_rank, num_workers, n)
                for col in range(left, right):
                    for row in range(top, bottom):
                        if worker_grid.get(row - top, col - left) is not None:
                            main_grid.set(row, col, worker_grid.get(row - top, col - left).copy())


    else: # worker

        worker_grid = comm.recv(source=0)

        # process the rounds
        for round in range(num_rounds_per_wave):

            # 1) movement phase

            air_unit_coords = list(worker_grid.air_units.keys())

            # get extra information from nearby processors if needed
            extra_left_needed = 0
            extra_right_needed = 0
            extra_top_needed = 0
            extra_bottom_needed = 0

            for coord in air_unit_coords:
                extra_left_needed = max(extra_left_needed, 3 - coord[1])
                extra_right_needed = max(extra_right_needed, 3 - (n - 1 - coord[1]))
                extra_top_needed = max(extra_top_needed, 3 - coord[0])
                extra_bottom_needed = max(extra_bottom_needed, 3 - (n - 1 - coord[0]))

            process_row = rank // num_workers_per_row
            process_col = rank % num_workers_per_row

            # process communication round
            # every process sends its grid information to all neighbors to avoid unnecessary waiting at either send or recv
            
            data = [worker_grid for _ in range(8)]
            incoming_data = communicate(data, rank, num_workers, comm)
            extended_grid = merge_grids(worker_grid, incoming_data, rank, num_workers, n)

            # 1. Movement Phase (Air Units)
            air_unit_coords = list(extended_grid.air_units.keys())
            movement_decisions = handle_air_movement(extended_grid, air_unit_coords, extended_grid.n)
            





            # decide where air units go, if they collide, merge



            # 2) action phase (units either attack or skip, attacks are buffered)


            # 3) resolution phase (apply the buffered attacks)

                # if a unit attacked by fire units dies, every fire unit gets a buff


            # 4) healing phase
                # add "did_attack" parameter to units, if a unit didn't attack, heal



        # reset the attack of fire units

        # clone waters to appropriate places

        # send data back to manager











    # cleanup
    MPI.Finalize()
    input_file.close()
    output_file.close()