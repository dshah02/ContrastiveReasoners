import random
import itertools
import operator
import sys
import time
import json

# Use a small epsilon for floating-point comparisons
EPSILON = 1e-6

def find_solution(numbers, target):
    """
    Tries to find ONE arithmetic expression that equals the target value
    using the given numbers. Uses +, -, *, / and parentheses.
    Each number must be used exactly once.

    Args:
        numbers (list): A list of four numbers (int or float).
        target (float or int): The target value to reach.

    Returns:
        str: A string representing the expression if a solution is found,
             otherwise None.
    """
    # Input validation (basic)
    if not isinstance(numbers, list) or len(numbers) != 4:
        # print(f"Debug: Invalid numbers input to find_solution: {numbers}")
        return None
    if not isinstance(target, (int, float)):
        # print(f"Debug: Invalid target input to find_solution: {target}")
        return None

    try:
        # Convert numbers to (string_representation, value) pairs for expression building
        num_pairs = [(str(n), float(n)) for n in numbers]
        # Use a recursive helper function, passing the target
        solution = _find_solution_recursive_helper(num_pairs, float(target))
        return solution
    except Exception as e:
        # print(f"Debug: Error during find_solution setup or recursion: {e}")
        return None


def _find_solution_recursive_helper(num_pairs, target):
    """Recursive helper to find a solution expression for the target value."""
    # Base case: one number left
    if len(num_pairs) == 1:
        expr_str, value = num_pairs[0]
        # Check if the final value is close enough to the target
        if abs(value - target) < EPSILON:
            return expr_str  # Solution found!
        else:
            return None      # This path doesn't lead to the target

    # Recursive step: try combining every pair of numbers
    for i in range(len(num_pairs)):
        for j in range(i + 1, len(num_pairs)):
            # Ensure indices are valid (should always be if logic is correct)
            if i >= len(num_pairs) or j >= len(num_pairs):
                # print(f"Debug: Index out of bounds in recursive helper. i={i}, j={j}, len={len(num_pairs)}")
                continue # Should not happen

            a_str, a_val = num_pairs[i]
            b_str, b_val = num_pairs[j]

            remaining_pairs = [num_pairs[k] for k in range(len(num_pairs)) if k != i and k != j]

            possible_ops = [
                ('+', f"({a_str} + {b_str})", a_val + b_val),
                ('-', f"({a_str} - {b_str})", a_val - b_val),
                ('*', f"({a_str} * {b_str})", a_val * b_val),
                ('-', f"({b_str} - {a_str})", b_val - a_val), # Order matters
            ]
            # Division a / b
            if abs(b_val) > EPSILON:
                possible_ops.append(('/', f"({a_str} / {b_str})", a_val / b_val))
            # Division b / a
            if abs(a_val) > EPSILON:
                possible_ops.append(('/', f"({b_str} / {a_str})", b_val / a_val))

            for op_sym, new_expr_str, new_val in possible_ops:
                # Recurse with the remaining numbers plus the new combined result, passing target
                solution = _find_solution_recursive_helper(remaining_pairs + [(new_expr_str, new_val)], target)
                if solution:
                    return solution # Propagate solution upwards

    return None # No solution found at this level

def generate_random_target_games(num_games_to_find=100, min_val=1, max_val=9, target_min=20, target_max=100, require_integers=True):
    """
    Generates a specified number of game puzzles where *each game* has a
    *randomly chosen target* within the specified range [target_min, target_max].

    Args:
        num_games_to_find (int): The target number of solvable games to find.
        min_val (int): The minimum value for the puzzle numbers (inclusive).
        max_val (int): The maximum value for the puzzle numbers (inclusive).
        target_min (int): The minimum value for the random target (inclusive).
        target_max (int): The maximum value for the random target (inclusive).
        require_integers (bool): If True, only uses integers in the generated puzzles.

    Returns:
        list: A list of dictionaries, where each dictionary represents a found game
              containing 'numbers' (list), 'target' (number), and 'solution' (str).
              Returns fewer than num_games_to_find if generation times out.
    """
    if min_val > max_val:
        print("Warning: min_val cannot be greater than max_val. No games generated.")
        return []
    if target_min > target_max:
        print("Warning: target_min cannot be greater than target_max. No games generated.")
        return []

    valid_games_data = []
    games_found_count = 0
    attempts = 0
    # Safety limit - might need more attempts if targets make solutions rare
    # Increase multiplier compared to fixed target generation
    max_attempts = num_games_to_find * 2500 + 100000 # Increased safety margin

    print(f"\nGenerating {num_games_to_find} game(s)...")
    print(f"  - Puzzle numbers range: {min_val}-{max_val}")
    print(f"  - Random target range for each game: {target_min}-{target_max}")

    start_time = time.time()

    while games_found_count < num_games_to_find and attempts < max_attempts:
        attempts += 1

        # 1. Generate a random target for THIS attempt
        current_target = random.randint(target_min, target_max)

        # 2. Generate 4 random numbers for the puzzle
        if require_integers:
            current_numbers = [random.randint(min_val, max_val) for _ in range(4)]
        else:
            # Example for floats (adjust range/precision as needed)
            current_numbers = [round(random.uniform(min_val, max_val), 1) for _ in range(4)]

        # 3. Try to find a solution for these numbers and THIS target
        solution = find_solution(current_numbers, current_target)

        # Optional: Print progress more frequently
        # if attempts % 500 == 0:
        #    print(f"  Attempt {attempts}, Found {games_found_count}/{num_games_to_find}...")

        if solution:
            # Found a valid game for this specific number/target pair!
            games_found_count += 1
            game_info = {
                "numbers": current_numbers,
                "target": current_target,
                "solution": solution
            }
            valid_games_data.append(game_info)
            # Optional: Print progress when a game is found
            print(f"  Found game {games_found_count}/{num_games_to_find} (Target: {current_target}, Nums: {current_numbers})")


    # --- Loop finished ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Generation Complete ---")
    print(f"Found {games_found_count} solvable games.")
    print(f"Total attempts: {attempts}")
    print(f"Duration: {duration:.2f} seconds")

    if attempts >= max_attempts and games_found_count < num_games_to_find:
        print(f"\nWarning: Reached max attempts ({max_attempts}) before finding all {num_games_to_find} games.")
        print(f"         Consider increasing max_attempts or checking target/number ranges if this happens often.")
    elif games_found_count == num_games_to_find:
        print(f"\nSuccessfully generated {games_found_count} games.")

    return valid_games_data

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    TARGET_MIN_RANGE = 20
    TARGET_MAX_RANGE = 100
    PUZZLE_NUM_MIN = 1
    PUZZLE_NUM_MAX = 9 # Standard range

    # while True:
    #     try:
    #         num_games_input = input(f"How many games to generate? (Each with a random target {TARGET_MIN_RANGE}-{TARGET_MAX_RANGE}): ")
    #         GAMES_TO_GENERATE = int(num_games_input)
    #         if GAMES_TO_GENERATE <= 0:
    #             print("Please enter a positive number of games.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Invalid input. Please enter a whole number.")
    GAMES_TO_GENERATE = 10000

    default_filename = f"random_target_games_{GAMES_TO_GENERATE}.json"
    # filename_input = input(f"Enter the output filename (press Enter for '{default_filename}'): ")
    OUTPUT_FILENAME = default_filename
    # if not OUTPUT_FILENAME.lower().endswith('.json'):
    #     OUTPUT_FILENAME += '.json'


    # --- Generate Games ---
    generated_games = generate_random_target_games(
        num_games_to_find=GAMES_TO_GENERATE,
        min_val=PUZZLE_NUM_MIN,
        max_val=PUZZLE_NUM_MAX,
        target_min=TARGET_MIN_RANGE,
        target_max=TARGET_MAX_RANGE,
        require_integers=True
    )

    # --- Save to JSON ---
    if generated_games:
        print(f"\nSaving {len(generated_games)} found games to '{OUTPUT_FILENAME}'...")
        try:
            with open(OUTPUT_FILENAME, 'w') as f:
                json.dump(generated_games, f, indent=4) # indent=4 makes the file readable
            print(f"Successfully saved games.")
        except IOError as e:
            print(f"\nError: Could not write to file '{OUTPUT_FILENAME}'. {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during JSON saving: {e}")
    else:
        print("\nNo games were generated (or found within max attempts), so no file was saved.")

    # --- Optional: Test Solver Separately ---
    print("\n--- Testing Solver Separately ---")
    test_numbers = [7, 5, 2, 1]
    test_target = random.randint(TARGET_MIN_RANGE, TARGET_MAX_RANGE) # Test with a random target too
    print(f"Numbers: {test_numbers}, Target: {test_target}")
    solution_test = find_solution(test_numbers, test_target)
    if solution_test:
        print(f"Found solution: {solution_test}")
        try:
            # Simple verification
            result = eval(solution_test)
            print(f"Verification (eval): {result}")
        except Exception as e:
            print(f"Could not verify solution using eval: {e}")
    else:
        print(f"No solution found by the solver for target {test_target}.")