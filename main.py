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
        self.attack_directions = [(1, 0), (0, 1), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)] # not exact, scale will be applied in handle_special_skills function
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

        # map from (x, y) to unit pointer
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
        return self.grid[i][j]

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

    worker_x = worker_rank % num_workers
    worker_y = worker_rank // num_workers

    left_boundary = worker_x * n
    right_boundary = (worker_x + 1) * n  # exclusive
    top_boundary = worker_y * n
    bottom_boundary = (worker_y + 1) * n  # exclusive

    return left_boundary, right_boundary, top_boundary, bottom_boundary

def send_grid_to_neighbors(grid, rank, num_workers, comm):
    """
    :param rank: the rank of the process
    :param num_workers: the total number of workers
    :param n: the number of lines each worker will be responsible
    :param grid: the grid that will be sent to neighbors
    :param comm: the communicator object
    :return: None
    """

    if not rank % num_workers_per_row == 0:  # check if rightmost process
        comm.send(grid, dest=rank + 1)
    if not rank % num_workers_per_row == 1:  # check if leftmost process
        comm.send(grid, dest=rank - 1)
    if not rank > num_workers - num_workers_per_row:
        comm.send(grid, dest=rank + num_workers_per_row)
    if not rank <= num_workers_per_row:
        comm.send(grid, dest=rank - num_workers_per_row)

def receive_grid_from_neighbors(rank, num_workers, n, comm):
    """
    :param rank: the rank of the process
    :param num_workers: the total number of workers
    :param n: the number of lines each worker will be responsible
    :param comm: the communicator object
    :return: (left_extra_grid, right_extra_grid, top_extra_grid, bottom_extra_grid) the extra information from neighbors
    """

    left_extra_grid = None
    right_extra_grid = None
    top_extra_grid = None
    bottom_extra_grid = None

    if not rank % num_workers_per_row == 0:  # check if rightmost process
        right_extra_grid = comm.recv(source=rank + 1)
    if not rank % num_workers_per_row == 1:  # check if leftmost process
        left_extra_grid = comm.recv(source=rank - 1)
    if not rank > num_workers - num_workers_per_row:
        bottom_extra_grid = comm.recv(source=rank + num_workers_per_row)
    if not rank <= num_workers_per_row:
        top_extra_grid = comm.recv(source=rank - num_workers_per_row)

    return left_extra_grid, right_extra_grid, top_extra_grid, bottom_extra_grid

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

            process_row = umut
            process_col = umut

            # process communication round
            # every process sends its grid information to all neighbors to avoid unnecessary waiting at either send or recv
            # unused extra info is then set to None
            if (process_row + process_col) % 2 == 0: # receive first, then send

                left_extra_grid, right_extra_grid, top_extra_grid, bottom_extra_grid = receive_grid_from_neighbors(rank, num_workers, n, comm)
                send_grid_to_neighbors(worker_grid, rank, num_workers, comm)

            else: # send first, then receive

                send_grid_to_neighbors(worker_grid, rank, num_workers, comm)
                left_extra_grid, right_extra_grid, top_extra_grid, bottom_extra_grid = receive_grid_from_neighbors(rank, num_workers, n, comm)

            if extra_left_needed == 0:
                left_extra_grid = None
            if extra_right_needed == 0:
                right_extra_grid = None
            if extra_top_needed == 0:
                top_extra_grid = None
            if extra_bottom_needed == 0:
                bottom_extra_grid = None





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