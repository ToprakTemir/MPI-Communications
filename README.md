# CMPE 300 - Analysis of Algorithms Project 2
Abdullah Umut Hamzaoğulları

Bora Toprak Temir

## Compilation and Execution Instructions

1. **Dependencies:**
   - Ensure Python 3.x is installed on your system.
   - Required libraries include `mpi4py`. Install them using:
     ```bash
     pip install mpi4py
     ```

2. **Compilation:**
   - No compilation is required as the project is implemented in Python.

3. **Execution:**
   - To run the main script, use the following command:
     ```bash
     mpiexec -n [P] python main.py <input.txt> <output.txt>
     ```
     Replace `[P]` with the number of processes you wish to use.

4. **Testing:**
   - Ensure the input file is correctly formatted and placed in the appropriate directory.
   - Outputs will be saved in the specified output file.

## Known Issues or Assumptions

### Known Issues:
- There are no known issues in the current implementation.

### Assumptions:
- The grid size $N$ is divisible by $\sqrt{P - 1}$, where $P$ is the total number of processes.
- The input file format is valid and adheres to the specifications provided in the project description.
- The execution environment has sufficient computational resources to handle the simulation.

## File Structure

- **`main.py`:** The main program file containing the implementation.
- **Input File:** Should be formatted as specified in the project description.
- **Output File:** Contains the final results of the simulation.

## Additional Information

- The program uses MPI for parallel computation. Ensure `mpiexec` is installed and configured in your environment.
- The communication model avoids deadlocks and ensures proper synchronization between processes.

---