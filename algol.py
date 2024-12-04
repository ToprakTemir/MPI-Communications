import mpi4py.MPI as MPI
import sys
from enum import Enum

# setup the MPI
if len(sys.argv) != 3:
    print("Usage: mpiexec -n [P] main.py <input.txt> <output.txt>")
    sys.exit(1)

MPI.Init()

comm = MPI.COMM_WORLD
manager = comm.Get_rank() == 0
num_workers = comm.Get_size() - 1

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

input_file = open(input_file_name, "r")
output_file = open(output_file_name, "w")

first_line = input_file.readline().strip().split(" ")

N = int(first_line[0])
num_waves = int(first_line[1])
num_units_per_wave = int(first_line[2])
num_rounds_per_wave = int(first_line[3])

class Element(Enum):
    EARTH = 0
    FIRE = 1
    WATER = 2
    AIR = 3


class Unit:
    def __init__(self, type, health, attack, heal_rate):
        self.type = type
        self.health = health
        self.attack = attack
        self.heal_rate = heal_rate

class Earth(Unit):
    def __init__(self):
        super().__init__("earth", 18, 2, 3)
        attack_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class Fire(Unit):
    def __init__(self):
        super().__init__("fire", 12, 4, 1)
        attack_directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]

class Water(Unit):
    def __init__(self):
        super().__init__("water", 14, 3, 2)
        attack_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

class Air(Unit):
    def __init__(self):
        super().__init__("air", 10, 2, 2)
        attack_directions = [(1, 0), (0, 1), (0, -1), (-1, 0)] # not exact, scale will be applied in handle_special_skills function


# TODO
def handle_special_skills():
    pass


# bence kalan detayları her processor kendi listesinin içinde (ya da birkaç listede) tutabilir

grid_element_list = [[{} for i in range(N)] for j in range(N)] # every pixel is a dictionary

# placeholder for real loop
for wave in range(num_waves):
    pass




# cleanup
MPI.Finalize()