"""
Author: Tom Helmuth
Description: Contains the VacuumEnvironment class, which
is used to experiment with agents and search methods.
"""

# Changes in direction based on letter.
DIRECTIONS = {"U" : (-1, 0),
              "D" : (1, 0),
              "L" : (0, -1),
              "R" : (0, 1)}

# The possible contents of a cell in the environment
CELL_CONTENTS = ["Empty", "Agent", "Obstacle", "Dirt"]

class VacuumEnvironment:
    """The environment in which your vacuum agents will clean.
    rows and cols define the area in which the agent can move.
    These are implicitly surrounded by obstacles, so that agents
    cannot move outside of the grid."""

    def __init__(self, rows, cols, agent_loc, clean_cells=[], obstacles=[]):
        """Parameters:
        rows and cols - the numbers of rows and columns of the environment.
        agent_loc - a tuple such as (3,0) that indicates the location of the agent.
        clean_cells - a list of tuples of locations, each of which is clean to start. Any cell not listed here or in obstacles starts out dirty.
        obstacles - a list of tuples of locations, each of which signifies the location of an obstacle."""

        self.rows = rows
        self.cols = cols

        # Tuple of the form (row, col), the location of the agent
        self.agent_loc = agent_loc

        # Grid of cells referenced by [row][col]
        self.grid = []
        for _ in range(self.rows):
            row = ["Dirt" for _ in range(self.cols)]
            self.grid.append(row)

        # Set clean cells
        for loc in clean_cells:
            self[loc] = "Empty"

        # Set obstacle cells
        for loc in obstacles:
            self[loc] = "Obstacle"

        # Make sure agent doesn't start on obstacle
        if self[self.agent_loc] == "Obstacle":
            raise ValueError("Agent cannot start on an obstacle.")
        else:
            self[self.agent_loc] = "Agent"

    def __getitem__(self, loc):
        """Gets the cell at loc, which is a tuple.
        This allows us to reference a VacuumEnvironment object by a location tuple
        with square brackets. For example, if ve is a VacuumEnvironment,
        then ve[(2,1)] would return the cell contents at location (2, 1)."""
        if not self.location_in_bounds(loc):
            raise IndexError("Location is out of bounds of the environment.")
        return self.grid[loc[0]][loc[1]]

    def __setitem__(self, loc, new_cell):
        """Sets the cell at loc to new_cell."""
        self.grid[loc[0]][loc[1]] = new_cell

    def location_in_bounds(self, loc):
        """Returns true if tuple loc is within the bounds of the environment."""
        (r, c) = loc
        return r >= 0 and c >= 0 and r < self.rows and c < self.cols

    def agent_can_enter(self, loc):
        """Returns true if tuple loc enterable by the agent, i.e. doesn't
        have an obstacle. Assumes location is within the grid of the
        environment."""
        return self[loc] != "Obstacle"

    def move_agent(self, direction):
        """Moves the agent in the direction, if possible.
        direction must be a string, the first character of which must be,
        when uppercased, one of U (up), D (down), L (left), R (right)."""

        dir = direction[0].upper()

        if dir not in "UDLR":
            raise ValueError("direction must be string whose first character is U, D, L, or R, but was instead" + direction)

        # Get new location
        (r, c) = self.agent_loc
        (dr, dc) = DIRECTIONS[dir]
        new_loc = (r + dr, c + dc)

        # Check for bounds of environment
        if not self.location_in_bounds(new_loc):
            return

        # Check for obstacles
        if not self.agent_can_enter(new_loc):
            return

        # Set new location
        self[self.agent_loc] = "Empty"
        self.agent_loc = new_loc
        self[self.agent_loc] = "Agent"
