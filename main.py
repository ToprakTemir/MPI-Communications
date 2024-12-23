import mpi4py.MPI as MPI
import sys


# unit stats
EARTH_HP = 18
EARTH_ATTACK = 2
EARTH_HEAL_RATE = 3
EARTH_ATTACK_DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

FIRE_HP = 12
FIRE_ATTACK = 4
FIRE_HEAL_RATE = 1
FIRE_ATTACK_DIRECTIONS = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

WATER_HP = 14
WATER_ATTACK = 3
WATER_HEAL_RATE = 2
WATER_ATTACK_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

AIR_HP = 10
AIR_ATTACK = 2
AIR_HEAL_RATE = 2
AIR_ATTACK_DIRECTIONS = [(1, 0), (0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

class Unit:
    def __init__(self, type, health, attack, heal_rate, attack_directions):
        self.type = type
        self.health = health
        self.attack = attack
        self.heal_rate = heal_rate
        self.attack_directions = attack_directions

        self.damage_to_be_taken = 0
        self.did_attack = False
        self.did_kill = False


def init_earth():
    return Unit("E", 18, 2, 3, [(0, 1), (1, 0), (0, -1), (-1, 0)])

def init_fire():
    return Unit("F", 12, 4, 1, [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)])

def init_water():
    return Unit("W", 14, 3, 2, [(1, 1), (1, -1), (-1, 1), (-1, -1)])

def init_air():
    attack_directions = [(1, 0), (0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    return Unit("A", 10, 2, 2, attack_directions)

def init_empty():
    return Unit("Empty", -1, -1, -1, [])

EMPTY_UNIT = init_empty()

# processes will have their own grid, and they will
class Grid:
    def __init__(self, n):
        self.n = n
        self.grid = [[EMPTY_UNIT for _ in range(n)] for _ in range(n)]
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
                if self.grid[i][j].type != "Empty":
                    ret += self.grid[i][j].type + " "
                else:
                    ret += ". "
            ret += "\n"
        return ret

    def get(self, i, j):
        if 0 <= i < self.n and 0 <= j < self.n:
            return self.grid[i][j]
        else:
            raise ValueError("Index out of bounds of the grid.")

    def set(self, i, j, pixel):
        self.grid[i][j] = pixel

        if pixel is None:
            raise ValueError("Pixel cannot be None")

        if pixel.type == "Empty":
            if pixel.type == "A":
                self.num_air += 1
                self.air_units[(i, j)] = pixel.unit
            elif pixel.type == "F":
                self.num_fire += 1
                self.fire_units[(i, j)] = pixel.unit
            elif pixel.type == "W":
                self.num_water += 1
                self.water_units[(i, j)] = pixel.unit
            elif pixel.type == "E":
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
        return init_earth()
    elif unit_type == "F":
        return init_fire()
    elif unit_type == "W":
        return init_water()
    elif unit_type == "A":
        return init_air()

def get_worker_borders(worker_rank, num_workers, n):
    """
    :param n: the number of lines each worker will be responsible
    :return: (left, right, top, bottom) boundary indexes of the area that the worker will be responsible for. \n
    all boundaries are inclusive

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
    wrank = rank - 1  # only the worker ranks are considered

    if not wrank % num_workers_per_row == 0:  # check if leftmost process
        print(f"sending from {rank} to {rank - 1}", flush=True)
        comm.send(data[0], dest=rank - 1)

    if not wrank % num_workers_per_row == num_workers_per_row - 1:  # check if rightmost process
        print(f"sending from {rank} to {rank + 1}", flush=True)
        comm.send(data[1], dest=rank + 1)
    

def send_to_vertical_neighbors(data, rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    wrank = rank - 1  # only the worker ranks are considered

    if not wrank < num_workers_per_row:
        print(f"sending from {rank} to {rank - num_workers_per_row}", flush=True)
        comm.send(data[0], dest=rank - num_workers_per_row)
    if not wrank >= num_workers - num_workers_per_row:
        print(f"sending from {rank} to {rank + num_workers_per_row}", flush=True)
        comm.send(data[1], dest=rank + num_workers_per_row)
    

def send_to_cross_neighbors(data, rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    wrank = rank - 1  # only the worker ranks are considered

    if not wrank % num_workers_per_row == 0 and not wrank < num_workers_per_row:
        print(f"sending from {rank} to {rank - num_workers_per_row - 1}", flush=True)
        comm.send(data[0], dest=rank - num_workers_per_row - 1)
    if not wrank % num_workers_per_row == num_workers_per_row - 1 and not wrank < num_workers_per_row:
        print(f"sending from {rank} to {rank - num_workers_per_row + 1}", flush=True)
        comm.send(data[1], dest=rank - num_workers_per_row + 1)
    if not wrank % num_workers_per_row == num_workers_per_row - 1 and not wrank >= num_workers - num_workers_per_row:
        print(f"sending from {rank} to {rank + num_workers_per_row + 1}", flush=True)
        comm.send(data[2], dest=rank + num_workers_per_row + 1)
    if not wrank % num_workers_per_row == 0 and not wrank >= num_workers - num_workers_per_row:
        print(f"sending from {rank} to {rank + num_workers_per_row - 1}", flush=True)
        comm.send(data[3], dest=rank + num_workers_per_row - 1)
    
    


def receive_from_horizontal_neighbors(rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    wrank = rank - 1  # only the worker ranks are considered

    left_data = None
    right_data = None

    if not wrank % num_workers_per_row == num_workers_per_row - 1:  # check if rightmost process
        print(f"waiting from {rank + 1} for {rank}", flush=True)
        right_data = comm.recv(source=rank + 1)
    if not wrank % num_workers_per_row == 0:  # check if leftmost process
        print(f"waiting from {rank - 1} for {rank}", flush=True)
        left_data = comm.recv(source=rank - 1)

    return left_data, right_data

def receive_from_vertical_neighbors(rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    wrank = rank - 1  # only the worker ranks are considered

    top_data = None
    bottom_data = None

    if not wrank >= num_workers - num_workers_per_row:
        print(f"waiting from {rank + num_workers_per_row} for {rank}", flush=True)
        bottom_data = comm.recv(source=rank + num_workers_per_row)
    if not wrank < num_workers_per_row:
        print(f"waiting from {rank - num_workers_per_row} for {rank}", flush=True)
        top_data = comm.recv(source=rank - num_workers_per_row)

    return top_data, bottom_data

def receive_from_cross_neighbors(rank, num_workers, comm):
    num_workers_per_row = int(num_workers ** 0.5)
    wrank = rank - 1  # only the worker ranks are considered

    top_left_data = None
    top_right_data = None
    bottom_left_data = None
    bottom_right_data = None

    if not wrank % num_workers_per_row == num_workers_per_row - 1 and not wrank >= num_workers - num_workers_per_row:
        print(f"waiting from {rank + num_workers_per_row + 1} for {rank}", flush=True)
        bottom_right_data = comm.recv(source=rank + num_workers_per_row + 1)
    if not wrank % num_workers_per_row == 0 and not wrank < num_workers_per_row:
        print(f"waiting from {rank - num_workers_per_row - 1} for {rank}", flush=True)
        top_left_data = comm.recv(source=rank - num_workers_per_row - 1)
    if not wrank % num_workers_per_row == num_workers_per_row - 1 and not wrank < num_workers_per_row:
        print(f"waiting from {rank - num_workers_per_row + 1} for {rank}", flush=True)
        top_right_data = comm.recv(source=rank - num_workers_per_row + 1)
    if not wrank % num_workers_per_row == 0 and not wrank >= num_workers - num_workers_per_row:
        print(f"waiting from {rank + num_workers_per_row - 1} for {rank}", flush=True)
        bottom_left_data = comm.recv(source=rank + num_workers_per_row - 1)

    return top_left_data, top_right_data, bottom_left_data, bottom_right_data


def point_letter(row, column):
    l = ["AB", "CD"]
    return l[row%2][column%2]

def send_data_to_neighbors(data, rank, num_workers, comm):
    send_to_cross_neighbors([data[0], data[2], data[5], data[7]], rank, num_workers, comm)
    send_to_horizontal_neighbors([data[3], data[4]], rank, num_workers, comm)
    send_to_vertical_neighbors([data[1], data[6]], rank, num_workers, comm)
    

def communicate(data, rank, num_workers, comm):
    """
    :param data: the grid that will be sent to neighbors, ordered as [upperleft, up, upperright, left, right, lowerleft, down, lowerright]
    :param rank: the rank of the process
    :param num_workers: the total number of workers
    :param comm: the communicator object
    :return: the received data from neighbors, ordered as [upperleft, up, upperright, left, right, lowerleft, down, lowerright]
    """

    num_workers_per_row = int(num_workers ** 0.5)
    process_row = (rank-1) // num_workers_per_row
    process_col = (rank-1) % num_workers_per_row

    print((process_row, process_col))
    print(point_letter(process_row, process_col))

    if point_letter(process_row, process_col) == "A":

        send_to_horizontal_neighbors([data[3], data[4]], rank, num_workers, comm)
        send_to_vertical_neighbors([data[1], data[6]], rank, num_workers, comm)
        send_to_cross_neighbors([data[0], data[2], data[5], data[7]], rank, num_workers, comm)

        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)
        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)

    elif point_letter(process_row, process_col) == "B":
        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)

        send_to_horizontal_neighbors([data[3], data[4]], rank, num_workers, comm)
        send_to_cross_neighbors([data[0], data[2], data[5], data[7]], rank, num_workers, comm)
        send_to_vertical_neighbors([data[1], data[6]], rank, num_workers, comm)

        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)

    elif point_letter(process_row, process_col) == "C":
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)
        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)

        send_to_vertical_neighbors([data[1], data[6]], rank, num_workers, comm)
        send_to_cross_neighbors([data[0], data[2], data[5], data[7]], rank, num_workers, comm)
        send_to_horizontal_neighbors([data[3], data[4]], rank, num_workers, comm)

        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)

    elif point_letter(process_row, process_col) == "D":
        cross_neighbor_data = receive_from_cross_neighbors(rank, num_workers, comm)
        vertical_neighbor_data = receive_from_vertical_neighbors(rank, num_workers, comm)
        horizontal_neighbor_data = receive_from_horizontal_neighbors(rank, num_workers, comm)

        send_to_cross_neighbors([data[0], data[2], data[5], data[7]], rank, num_workers, comm)
        send_to_vertical_neighbors([data[1], data[6]], rank, num_workers, comm)
        send_to_horizontal_neighbors([data[3], data[4]], rank, num_workers, comm)

    return cross_neighbor_data[0], vertical_neighbor_data[0], cross_neighbor_data[1], horizontal_neighbor_data[0], horizontal_neighbor_data[1], cross_neighbor_data[2], vertical_neighbor_data[1], cross_neighbor_data[3] 

class ExtendedGrid:
    """
    A class to represent the grid that includes the grids of the neighboring processors
    """

    def __init__(self, own_grid, nearby_grids, rank, num_workers, n):
        self.grids = [nearby_grids[0], nearby_grids[1], nearby_grids[2], nearby_grids[3],
                      own_grid,
                      nearby_grids[4], nearby_grids[5], nearby_grids[6], nearby_grids[7]]
        self.rank = rank
        self.num_workers = num_workers
        self.n = n

    def get(self, row, col):
        """
        Takes in i, j coordinates relative to the own grid and returns the corresponding pixel
        """

        n = self.n
        left = col < 0
        right = col >= n
        top = row < 0
        bottom = row >= n

        ret = None

        if left and top:
            ret = self.grids[0].get(row + n, col + n)
        elif right and top:
            ret = self.grids[2].get(row + n, col - n)
        elif top:
            ret = self.grids[1].get(row + n, col)
        elif left and bottom:
            ret = self.grids[5].get(row - n, col + n)
        elif right and bottom:
            ret = self.grids[7].get(row - n, col - n)
        elif bottom:
            ret = self.grids[6].get(row - n, col)
        elif left:
            ret = self.grids[3].get(row, col + n)
        elif right:
            ret = self.grids[4].get(row, col - n)

        return ret

def coords_relative_to_grid_index(row, col, grid_index, n):
    """
    converts the coordinates relative to the own grid to the coordinates relative to the grid_index-th neighbor grid
    """

    if grid_index == 0:
        return row - n, col - n
    elif grid_index == 1:
        return row - n, col
    elif grid_index == 2:
        return row - n, col + n
    elif grid_index == 3:
        return row, col - n
    elif grid_index == 4:
        return row, col
    elif grid_index == 5:
        return row + n, col - n
    elif grid_index == 6:
        return row + n, col
    elif grid_index == 7:
        return row + n, col + n
    else:
        return -1, -1


def get_grid_index(row, col, n):
    """
    what neighbor grid index does row and col (given rel. to own grid) belong to
    """

    left = col < 0
    right = col >= n
    top = row < 0
    bottom = row >= n

    if left and top:
        return 0
    elif right and top:
        return 2
    elif top:
        return 1
    elif left and bottom:
        return 5
    elif right and bottom:
        return 7
    elif bottom:
        return 6
    elif left:
        return 3
    elif right:
        return 4
    else:
        return -1

def handle_air_movement(extended_grid):
    """
    Handles the movement phase for Air units
    Returns a list of movement decisions (of owned air units) to be applied simultaneously
    """

    n = 3 * extended_grid.n

    air_unit_coords = extended_grid.grids[4].air_units.keys()

    movement_decisions = []  # List of (from_coord, to_coord) tuples
    
    for coord in air_unit_coords:
        best_position = coord
        max_attackable = count_attackable_enemies(extended_grid, coord)
        
        pixel = extended_grid.get(coord[0], coord[1])
        # Check all possible movement directions
        for dx, dy in pixel.unit.attack_directions:
            new_x = coord[0] + dx
            new_y = coord[1] + dy
            
            if extended_grid.get(new_x, new_y).type == "Empty":
                attackable = count_attackable_enemies(extended_grid, (new_x, new_y), n)
                
                if attackable > max_attackable:
                    max_attackable = attackable
                    best_position = (new_x, new_y)
                elif attackable == max_attackable and attackable > count_attackable_enemies(extended_grid, coord, n):
                    if new_x < best_position[0] or (new_x == best_position[0] and new_y < best_position[1]):
                        best_position = (new_x, new_y)
        
        if best_position != coord:
            movement_decisions.append((coord, best_position))
    
    return movement_decisions

def count_attackable_enemies(extended_grid, coord):
    """
    Counts how many enemy units an Air unit can attack from the given position
    """

    n = 3 * extended_grid.n

    count = 0
    pixel = extended_grid.get(coord[0], coord[1])
    if pixel.type == "Empty" or pixel.type != 'A':
        return 0
        
    # Check all directions including diagonals
    for drow, dcol in pixel.unit.attack_directions:
        row = coord[0] + drow
        col = coord[1] + dcol
        
        if 0 <= row < n and 0 <= col < n:
            if extended_grid.get(row, col).type != "Empty" and extended_grid.get(row, col).type != 'A':
                count += 1
            elif extended_grid.get(row, col).type == "Empty":
                row += drow
                col += dcol
                if 0 <= row < n and 0 <= col < n and extended_grid.get(row, col).type != "Empty" and extended_grid.get(row, col).type != 'A':
                    count += 1
                    
    return count

def merge_air_units(air1, air2):
    air1.attack += air2.attack
    air1.health = min(AIR_HP, air1.health + air2.health)
    return air1

def _debug_print_arrived(checkpoint):
    print(f"Reached checkpoint {checkpoint} with worker {rank}", flush=True)


def attack_inside_grid(grid, unit_coord, attack_coord):
    """
    output
    0 means tried attack coord is empty, \n
    1 means attack is successful, \n
    -1 means attack is not successful because the units are of the same type
    """

    if grid.get(attack_coord[0], attack_coord[1]).type == "Empty":
        return 0

    attacking_unit = grid.get(unit_coord[0], unit_coord[1])
    attacked_unit = grid.get(attack_coord[0], attack_coord[1])
    if attacked_unit.type == attacking_unit.type:
        return -1

    attacked_unit.damage_to_be_taken += attacking_unit.unit.attack
    return 1



if __name__ == "__main__":

    # check if the terminal input is correct
    if len(sys.argv) != 3:
        print("Usage: mpiexec -n [P] main.py <input.txt> <output.txt>")
        sys.exit(1)

    # set up the MPI
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # will be different for every process, 0 is the manager
    num_workers = comm.Get_size() - 1
    num_workers_per_row = int(num_workers ** 0.5)



    if rank == 0: # manager

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
        n = N // int(num_workers ** 0.5)  # the number of lines each worker will be responsible
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
                    i, j = coord.split() # x and y are swapped in the input file
                    i = int(i)
                    j = int(j)
                    if main_grid.get(i, j).type == "Empty":
                        main_grid.set(i, j, create_unit(unit_type))
                    else:
                        pass # Don't override existing units

            line_index += 5

            # 2) distribute each wave to workers

            _debug_print_arrived(0)

            for worker_rank in range(1, num_workers + 1):

                # these worker x, y values are the x, y values of processors in the processor-grid
                # the indexes start from 0, that's why I subtracted 1 from the worker_rank
                left, right, top, bottom = get_worker_borders(worker_rank, num_workers, n)

                worker_grid = Grid(n)

                for col in range(left, right):
                    for row in range(top, bottom):
                        if worker_grid.get(row - top, col - left).type != "Empty":
                            worker_grid.set(row - top, col - left, main_grid.get(row, col).copy())

                comm.send([worker_grid, N, n, num_rounds_per_wave], dest=worker_rank)


            # 3) collect the results from workers before the next wave

            for worker_rank in range(1, num_workers + 1):
                worker_grid = comm.recv(source=worker_rank)
                left, right, top, bottom = get_worker_borders(worker_rank, num_workers, n)
                for col in range(left, right):
                    for row in range(top, bottom):
                        if worker_grid.get(row - top, col - left).unit.t != "Empty":
                            main_grid.set(row, col, worker_grid.get(row - top, col - left).copy())


    else: # worker

        worker_grid, N, n, num_rounds_per_wave = comm.recv(source=0)

        # process the rounds
        for round in range(num_rounds_per_wave):

            # _debug_print_arrived(1.1)

            # 1) movement phase

            air_unit_coords = list(worker_grid.air_units.keys())

            # communication with neighbors
            data = [worker_grid for _ in range(8)]
            print("BR", flush=True)
            neighbor_grids = communicate(data, rank, num_workers, comm)
            print("UH", flush=True)
            extended_grid = ExtendedGrid(worker_grid, neighbor_grids, rank, num_workers, n)

            # 1. Movement Phase (Air Units)
            movement_decisions = handle_air_movement(extended_grid)

            airs_and_destinations_to_send = [[] for _ in range(8)]

            _debug_print_arrived(1.2)

            for air_coords, air_dest in movement_decisions:

               # handle inside if dest is in the own grid
                if 0 <= air_dest[0] < n and 0 <= air_dest[1] < n:
                    if worker_grid.get(air_dest[0], air_dest[1]).type == "Empty":
                        worker_grid.set(air_dest[0], air_dest[1], worker_grid.get(air_coords[0], air_coords[1]))
                    else:
                        # merge air units
                        new_unit = merge_air_units(
                            worker_grid.get(air_dest[0], air_dest[1]).unit,
                            worker_grid.get(air_coords[0], air_coords[1]).unit
                        )
                        worker_grid.set(air_dest[0], air_dest[1], new_unit)

                    worker_grid.set(air_coords[0], air_coords[1], EMPTY_UNIT)

                # send to the appropriate neighbor grid otherwise
                else:
                    grid_index = get_grid_index(air_dest[0], air_dest[1], n)
                    air_to_send = worker_grid.get(air_coords[0], air_coords[1])
                    air_dest = coords_relative_to_grid_index(air_dest[0], air_dest[1], grid_index, n)
                    airs_and_destinations_to_send[grid_index].append((air_to_send, air_dest))

                    worker_grid.set(air_coords[0], air_coords[1], EMPTY_UNIT)


            incoming_air_destinations = communicate(airs_and_destinations_to_send, rank, num_workers, comm)
            incoming_air_destinations = [x for x in incoming_air_destinations]

            for air_unit, air_dest in incoming_air_destinations:
                if worker_grid.get(air_dest[0], air_dest[1]).type == "Empty":
                    worker_grid.set(air_dest[0], air_dest[1], air_unit)
                else:
                    # merge air units
                    new_unit = merge_air_units(
                        worker_grid.get(air_dest[0], air_dest[1]).unit,
                        air_unit.unit
                    )
                    worker_grid.set(air_dest[0], air_dest[1], new_unit)

            # 2) action phase (units either attack or skip, attacks are buffered)

            _debug_print_arrived(2.0)

            # handling internal attacks and preparing the inter-process attack data to send to neighbors

            unit_lists = [
                worker_grid.air_units,
                worker_grid.fire_units,
                worker_grid.water_units,
                worker_grid.earth_units
            ]

            attacker_and_dest_to_send = [[] for _ in range(8)]
            for unit_of_single_type in unit_lists:
                for coord, unit in unit_of_single_type:

                    if unit.type == 'A':
                        for direction in unit.attack_directions:
                            dx, dy = direction
                            attack_coord = [coord[0] + dx, coord[1] + dy]
                            far_attack_coord = [coord[0] + 2*dx, coord[1] + 2*dy]
                            if 0 <= attack_coord[0] < n and 0 <= attack_coord[1] < n:

                                try_close_attack = attack_inside_grid(worker_grid, coord, attack_coord)
                                if try_close_attack == -1:
                                    continue
                                if try_close_attack == 1:
                                    unit.did_attack = True
                                    continue

                                if 0 <= far_attack_coord[0] < n and 0 <= far_attack_coord[1] < n:
                                    attack_inside_grid(worker_grid, coord, far_attack_coord)
                                else:
                                    grid_index = get_grid_index(far_attack_coord[0], far_attack_coord[1], n)
                                    attacker_and_dest_to_send[grid_index].append((unit, far_attack_coord))
                        continue

                    # below is non-air units
                    attack_coords = [[unit[0] + dx, unit[1] + dy] for dx, dy in unit.attack_directions]
                    for attack_coord in attack_coords:
                        if 0 <= attack_coord[0] < n and 0 <= attack_coord[1] < n:
                            attack_inside_grid(worker_grid, coord, attack_coord)
                        else:
                            grid_index = get_grid_index(attack_coord[0], attack_coord[1], n)
                            attacker_and_dest_to_send[grid_index].append((unit, attack_coord))

            _debug_print_arrived(2.1)

            incoming_attacks = communicate(attacker_and_dest_to_send, rank, num_workers, comm)
            incoming_attacks = [x for x in incoming_attacks]
            for attacker, coord in incoming_attacks:
                attacked_pixel = worker_grid.get(coord[0], coord[1])
                if attacked_pixel.type != "Empty" and attacked_pixel.type != attacker.type:
                    attacked_pixel.damage_to_be_taken += attacker.attack

            _debug_print_arrived(2.2)

            # apply damage
            for pixel in [unit_list for unit_list in unit_lists]:
                for coord, unit in pixel:
                    if unit.type == "E":
                        # halve damage
                        unit.damage_to_be_taken //= 2

                    # ALL THE MORE SHIT











            # 3) resolution phase (apply the buffered attacks)

                # if a unit attacked by fire units dies, every fire unit gets a buff


            # 4) healing phase
                # add "did_attack" parameter to units, if a unit didn't attack, heal



        # reset the attack of fire units

        # clone waters to appropriate places

        # send data back to manager

    if rank == 0:
        input_file.close()
        output_file.close()
    # cleanup