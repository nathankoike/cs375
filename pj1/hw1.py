"""
Author:         Nate Koike & Zhaosen Guo
Date:           2021/2/8
Description:    Implement an agent to clean a 2d environment randomly
"""

import vacuum_environment as ve
import random

# needed for optinal animation
import time


class VacuumEnvironment375(ve.VacuumEnvironment):
    """ an extension of the VacuumEnvironment class that adds string conversion
        and a new method that determines if the entire environment is clean. """

    def __init__(self, rows, cols, agent_loc, clean_cells=[], obstacles=[]):
        # just initialize the superclass since we don't need any extra data
        super(VacuumEnvironment375, self).__init__(rows, cols, agent_loc, clean_cells, obstacles)

    def __str__(self):
        """Print the envt with all features (obstacles, agent, and dirt). """
        # upper boundary
        s = "-" * (self.cols + 2) + "\n"

        # The character representations of things in the environment
        env_chars = {
            "Dirt": ".",
            "Empty": " ",
            "Obstacle": "#",
            "Agent": "@"
            }

        # Actual grid contents with new lines and left/right boundaries
        for r in self.grid:
            s += "|"
            for c in r:
                s += env_chars[c]
            s += "|\n"

        # lower boundary
        s += "-" * (self.cols + 2) + "\n"

        return s


    def __hash__(self):
        """ A way to hash the contents of the environment. """
        char_int = {
            "Dirt": "1",
            "Empty": "2",
            "Obstacle": "3",
            "Agent": "4"
            }
        int_str = ""

        for row in self.grid:
            for col in row:
                int_str += char_int[col]

        return int(int_str)


    def __eq__(self, other):
        """ Compare equality between two environments. """

        return self.grid == other.grid


    def all_clean(self):
        """ Return true if all the dirts are cleaned, false if not. """
        for r in self.grid:
            for c in r:
                if c == "Dirt":
                    return False
        return True

    def count_dirt(self):
        """ Return the total dirty squares in the environment. """
        count = 0
        if self.all_clean():
            return count

        for r in self.grid:
            for c in r:
                if c == "Dirt":
                    count += 1
        return count

    def return_surround(self):
        """ Return all the valid locations around (UDLR only) the agent. """
        # All the valid tiles around the agent
        tiles = []

        # Check every linear direction
        for dir in "UDRL":
            # Get new location
            (r, c) = self.agent_loc
            (dr, dc) = ve.DIRECTIONS[dir]
            new_loc = (r + dr, c + dc)

            # if the location is valid
            if (self.location_in_bounds(new_loc)):
                # add the location to the list
                tiles.append(self.grid[new_loc[0]][new_loc[1]])

        # Check every diagonal direction
        for dia_dir in [(1,1), (1,-1),(-1,-1),(-1,1)]:
            # Get new location
            (r, c) = self.agent_loc
            new_loc = (r + dia_dir[0], c + dia_dir[1])

            # if the location is valid
            if (self.location_in_bounds(new_loc)):
                # add the location to the list
                tiles.append(self.grid[new_loc[0]][new_loc[1]])

        return tiles

    def return_adjacent(self):
        """ Return all the valid locations that surronds the agent, (UDLR and
            diagonal) """
        # All the valid tiles around the agent
        tiles = []
        # Check every direction
        for dir in "UDRL":
            # Get new location
            (r, c) = self.agent_loc
            (dr, dc) = ve.DIRECTIONS[dir]
            new_loc = (r + dr, c + dc)

            # if the location is valid
            if (self.location_in_bounds(new_loc)):
                # add the location to the list
                tiles.append(self.grid[new_loc[0]][new_loc[1]])
        return tiles


def random_agent(vac_envt, animate=False):
    """ Moves the agent untill the envt is clean, printing at every
        step and present the total steps taken
        vac_envt: the environment for the agent to clean
        animate: an optional parameter that toggles animation of the agent """
    # the counter for how many steps the agent has taken
    step = 0

    # while the environment is not clean
    while not vac_envt.all_clean():
        # try to move in a random direction
        vac_envt.move_agent(random.choice(['U','D','L','R']))

        # increment the step counter
        step += 1

        # print the environment and the current step counter
        print(vac_envt)
        print("Current step:", step, "\n")

        # so you can actually observe what's going on
        if animate:
            time.sleep(0.01)

    # print the total number of steps
    print("Total steps taken:", step)


def random_agent_data(vac_envt):
    """ Almost identical to random_agent() function but elimates all printing
        and returns total steps taken. """
    step = 0
    while not vac_envt.all_clean():
        vac_envt.move_agent(random.choice(['U','D','L','R']))
        step += 1
    return step


def make_vacuum_environment(k):
    """ make 1 of 4 environments; which environment is determined by n
        k - the environment number (0 is the smallest in size). """
    if k < 1:
        return VacuumEnvironment375(3, 3, (0, 0), clean_cells=[], obstacles=[(1, 1)])

    if k < 2:
        return VacuumEnvironment375(4,4,(0, 0), clean_cells=[], obstacles=[(1, 2), (2, 2)])

    if k < 3:
        return VacuumEnvironment375(8, 8, (0, 0), clean_cells=[(0,1)], obstacles=[])

    else:
        return VacuumEnvironment375(9, 9, (0, 0), clean_cells=[], obstacles=[(
            7, 3), (6, 3), (5, 3), (5, 2), (5, 1), (5, 0), (7, 1), (8, 1)])


def env_test(n):
    """ Run n tests on the environments accessible through the
        make_vacuum_environment function """
    # for every accessible environment
    for i in range(4):
        # show the environment being tested
        print(make_vacuum_environment(i))

        # the minimum number of steps to clean the environment
        # this is just a really big number thats a placeholder for now, and
        # while it's theoretically possible that this is a minimum it's
        # practically impossible for that to happen
        min = 999999999999999999999999999

        # the maximum number of steps to clean the environment
        max = 0

        # the total number of steps taken during the testing
        total = 0

        # run n trials
        for _ in range(n):
            steps = random_agent_data(make_vacuum_environment(i))

            # check for a max
            if steps > max:
                max = steps

            # check for a min
            if steps < min:
                min = steps

            # add to the total number of steps
            total += steps

        # print the results nicely with clear separation between tests
        print("env.", i)
        print("min:", min, "\tmax:", max, "\tmean:", total / n)
        print("\n\n")


def main():
    """ The main function to generate a vacuum environment and related calls."""
    # Creating a simple 8x8 envt and use run the agent
    new_envt = VacuumEnvironment375(8, 8, (0, 0), clean_cells=[], obstacles=[])

    # run the agent on the 8x8 environment
    random_agent(new_envt)

    # Make an environment based on the desired parameter
    for i in range(4):
        random_agent(make_vacuum_environment(i), False)

    # Run and print results from the testing function for stats
    print("\n\n")
    env_test(1000)
