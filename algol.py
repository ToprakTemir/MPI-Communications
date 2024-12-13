import mpi4py.MPI as MPI
import sys

class Unit:
    def __init__(self, type, health, attack, heal_rate):
        self.type = type
        self.health = health
        self.attack = attack
        self.heal_rate = heal_rate

class Earth(Unit):
    def __init__(self):
        super().__init__("earth", 18, 2, 3)
        self.attack_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class Fire(Unit):
    def __init__(self):
        super().__init__("fire", 12, 4, 1)
        self.attack_directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

class Water(Unit):
    def __init__(self):
        super().__init__("water", 14, 3, 2)
        self.attack_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

class Air(Unit):
    def __init__(self):
        super().__init__("air", 10, 2, 2)
        self.attack_directions = [(1, 0), (0, 1), (0, -1), (-1, 0)] # not exact, scale will be applied in handle_special_skills function

class Pixel:
    def __init__(self, unit):
        self.unit = unit
        self.damage_to_be_taken = 0

    def __str__(self):
        return self.unit.type

    def copy(self):
        return Pixel(self.unit)


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


if __name__ == "__main__":

    # setup the MPI
    if len(sys.argv) != 3:
        print("Usage: mpiexec -n [P] main.py <input.txt> <output.txt>")
        sys.exit(1)

    MPI.Init()

    comm = MPI.COMM_WORLD
    is_manager = comm.Get_rank() == 0
    num_workers = comm.Get_size() - 1

    if is_manager:
        input_file_name = sys.argv[1]
        output_file_name = sys.argv[2]

        input_file = open(input_file_name, "r")
        output_file = open(output_file_name, "w")

        lines = input_file.readlines()
        first_line = lines[0].strip().split(" ")
        line_index = 1

        N = int(first_line[0])
        num_waves = int(first_line[1])
        num_units_per_faction_per_wave = int(first_line[2])
        num_rounds_per_wave = int(first_line[3])


    grid_element_list = [[None for i in range(N)] for j in range(N)]

    for wave in range(num_waves):

        # parse the file and get placements
        while not lines[line_index].startswith("Wave"):
            line_index += 1

        for i in range(1, 5):
            line = lines[line_index + i]
            unit_type = line[0]
            coordinates = line[2:].strip().split(",")

            for coord in coordinates:
                x, y = coord.split(" ")
                x = int(x)
                y = int(y)
                if grid_element_list[y][x] is None:
                    grid_element_list[y][x] = Pixel(create_unit(unit_type))
                else:
                    pass # Don't override existing units (CAN CHANGE)

        line_index += 5


        # distribute to workers

        # reset the attack of fire units before the first round

        # process the rounds

        # clone waters to appropriate places

        # send data back to manager



        for round in range(num_rounds_per_wave):
            pass


        # write output



    # cleanup
    MPI.Finalize()
    input_file.close()
    output_file.close()