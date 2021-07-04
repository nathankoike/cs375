import numpy as np

class Node:
    """ A node class to hold a list of children and their would-be heuristic
    evaluations """
    def __init__(self, h_val, children):
        self.h = h_val
        self.children = children

def mm(node, max_t):
    """ The standard Minimax algorithm """
    # If there are no children (i.e. terminal nodes)
    if node.children == []:
        return (node.h, None)

    # If it's MAX's turn
    if max_t:
        val = -np.inf

        # For every child
        for child in node.children:
            # The value of the move and the resulting state are the result of a
            # recursive call to the minimax algorithm
            move_val, move = mm(child, not max_t)

            # If the value of the recursive call is better than the current one
            if move_val > val:
                # Save the value and assign a new best result
                val = move_val
                best = move

        # Return the value of the best move along with the state resulting from
        # that move
        return val, best

    # It's MIN's turn
    else:
        val = np.inf

        # This is the same as MAX's turn but we want the value to be smaller
        for child in node.children:
            move_val, move = mm(child, not max_t)
            if move_val < val:
                val = move_val
                best = move
        return val, best

def abp(node, a, b, max_t):
    """ The standard Alpha-beta pruning algorithm """
    # If there are no children (i.e. terminal node)
    if node.children == []:
        return (node.h, None)

    # If it's MAX's turn
    if max_t:
        val = -np.inf

        # For every child
        for child in node.children:
            # Find the maximum value of the current value and the child's
            # heuristic evaluation
            val = max(val, abp(child, a, b, not max_t)[0])

            # Set alpha to be the max of the value and the current alpha
            a = max(val, a)

            # If alpha is greater than beta we can return
            if a >= b:
                return (val, node)

        return (val, node)

    # It's MIN's turn
    else:
        val = np.inf

        # This is the same as MAX's turn but we're minimizing and using beta
        for child in node.children:
            val = min(val, abp(child, a, b, not max_t)[0])
            b = min(val, b)
            if a >= b:
                return (val, node)

        return (val, node)

root = Node(-1, [
            Node(-1,[
                Node(-1,[
                    Node(-1, [
                        Node(41, []),
                        Node(11, [])]),
                    Node(-1, [
                        Node(9, []),
                        Node(37, [])])]),
                Node(-1,[
                    Node(-1, [
                        Node(52, []),
                        Node(48, [])]),
                    Node(-1, [
                        Node(20, []),
                        Node(30, [])])])]),
            Node(-1,[
                Node(-1,[
                    Node(-1, [
                        Node(10, []),
                        Node(27, [])]),
                    Node(-1, [
                        Node(9, []),
                        Node(37, [])])]),
                Node(-1,[
                    Node(-1, [
                        Node(50, []),
                        Node(36, [])]),
                    Node(-1, [
                        Node(25, []),
                        Node(3, [])])])])])

# print(abp(root, -np.inf, np.inf, True)[0])
print(mm(root, True)[0])
