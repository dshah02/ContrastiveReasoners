import itertools
from typing import List, Union, Tuple

# Define Number type for clarity, accommodating integers and potential floats from division
Number = Union[int, float]


def get_next_states(current_state: List[Number]) -> List[List[Number]]:
    """
    Generates all possible next states in the game 24 reachable in one step.

    A step consists of choosing two distinct numbers from the current state,
    performing one arithmetic operation (+, -, *, /) on them, and forming
    a new state with the result and the remaining numbers.

    Args:
        current_state: A list of numbers (int or float) representing the
                       current state. Expected length <= 4.

    Returns:
        A list of lists, where each inner list is a possible next state.
        Returns an empty list if the current state has fewer than 2 numbers.
        Note: Division by zero is handled by skipping that operation.
              Duplicate states resulting from different operations/pairs
              (e.g., 2+2=4 and 2*2=4) are included.
    """
    next_states: List[List[Number]] = []
    n = len(current_state)

    if n < 2:
        # Cannot perform an operation with less than 2 numbers
        return []

    # Get all indices of the current state list
    indices = list(range(n))

    # Iterate through all unique combinations of 2 *indices*
    # itertools.combinations ensures we pick each pair of numbers (by position)
    # exactly once and avoids picking the same number twice (i != j).
    for i, j in itertools.combinations(indices, 2):
        # Get the two numbers based on the chosen indices
        a = current_state[i]
        b = current_state[j]

        # Identify the remaining numbers (those not at index i or j)
        remaining: List[Number] = [current_state[k] for k in indices if k != i and k != j]

        # --- Calculate results of all possible operations ---
        possible_results: List[Number] = []

        # 1. Addition
        possible_results.append(a + b)

        # 2. Subtraction (order matters)
        possible_results.append(a - b)
        possible_results.append(b - a) # Include both orders

        # 3. Multiplication
        possible_results.append(a * b)

        # 4. Division (order matters, check for division by zero)
        if b != 0:
            possible_results.append(a / b)
        if a != 0:
            possible_results.append(b / a) # Include both orders

        # --- Create the new states ---
        # For each valid arithmetic result, form the new state list:
        # It contains the remaining numbers plus the result of the operation.
        for result in possible_results:
            new_state = remaining + [result]
            next_states.append(new_state)

    return next_states


def list_all_valid_states(states): # this function should take in the current states plus all past states to allow backtracking
    """
    List all valid states from a given list of states.

    Args:
        states: A list of states (each state is a list of numbers).

    Returns:
        A list of all valid states generated from the input states.
    """
    all_states = []
    for state in states:
        next_states = get_next_states(state)
        all_states.extend(next_states)
    return all_states

states = [[1,2], [3,4]] 
print(list_all_valid_states(states))



# # Example 1: Starting state [1, 2, 3, 4]
# state1 = [1, 2, 3, 4]
# next_states1 = get_next_states(state1)
# print(f"Current state: {state1}")
# print(f"Possible next states ({len(next_states1)}):")
# # Print a sample
# print(next_states1[:6], "...")
# # Example operations from state1:
# # Pick 1, 2: remaining [3, 4]. Operations: 1+2=3 -> [3,4,3], 1-2=-1 -> [3,4,-1], 2-1=1 -> [3,4,1], ...
# # Pick 3, 4: remaining [1, 2]. Operations: 3+4=7 -> [1,2,7], 3-4=-1 -> [1,2,-1], 4-3=1 -> [1,2,1], ...

# print("-" * 20)

# # Example 2: Intermediate state [3, 3, 4] (e.g., from 1+2=3 in the previous step)
# state2 = [3, 3, 4]
# next_states2 = get_next_states(state2)
# print(f"Current state: {state2}")
# print(f"Possible next states ({len(next_states2)}):")
# print(next_states2)
# # Example operations from state2:
# # Pick 3, 3 (indices 0, 1): remaining [4]. Operations: 3+3=6 -> [4,6], 3-3=0 -> [4,0], 3*3=9 -> [4,9], 3/3=1 -> [4,1]
# # Pick 3, 4 (indices 0, 2): remaining [3]. Operations: 3+4=7 -> [3,7], 3-4=-1 -> [3,-1], 4-3=1 -> [3,1], ...
# # Pick 3, 4 (indices 1, 2): remaining [3]. Operations: (same as above) -> [3,7], [3,-1], [3,1], ...

# print("-" * 20)

# # Example 3: State with 2 numbers [6, 4]
# state3 = [6, 4]
# next_states3 = get_next_states(state3)
# print(f"Current state: {state3}")
# print(f"Possible next states ({len(next_states3)}):")
# print(next_states3)
# # Example operations from state3:
# # Pick 6, 4: remaining []. Operations: 6+4=10 -> [10], 6-4=2 -> [2], 4-6=-2 -> [-2], 6*4=24 -> [24], ...

# print("-" * 20)

# # Example 4: State with 1 number [24] (Goal state or intermediate)
# state4 = [24]
# next_states4 = get_next_states(state4)
# print(f"Current state: {state4}")
# print(f"Possible next states ({len(next_states4)}):")
# print(next_states4) # Should be empty

# print("-" * 20)

# # Example 5: Empty state []
# state5 = []
# next_states5 = get_next_states(state5)
# print(f"Current state: {state5}")
# print(f"Possible next states ({len(next_states5)}):")
# print(next_states5) # Should be empty