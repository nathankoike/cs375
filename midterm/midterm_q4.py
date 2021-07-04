Depths = [5, 10, 25, 50, 100, 1000, 10000]
Branching = [1, 3, 10, 25, 50, 100, 1000, 10000]

def ids_calc(branch, depth):
    """ calculate how many nodes ids will expand """
    nodes = 0

    # for every depth up to the specified depth (depth starts at 1)
    for d in range(1, depth + 1):
        # at every depth, the number of nodes we expand is equal to the
        # branching factor raised to the current depth - 1
        nodes += branch ** d

    return nodes

def bfs_calc(branch, depth):
    """ calculate how many nodes bfs will expand """
    return branch ** depth

def main():
    for d in Depths:
        for b in Branching:
            ids = ids_calc(b, d)
            bfs = bfs_calc(b, d)

            print("branch:", b)
            print(" depth:", d)
            print((ids - bfs) / bfs)
            print()

if __name__ == "__main__":
    main()
