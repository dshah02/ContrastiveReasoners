SYSTEM_PROMPT = """
We will be playing the game 24. In this game you will start with 4 numbers from 1 to 10 and aim to generate a target number using each number once.
To generate the target number, you will combine the 4 starting numbers using +, -, *, and /. 
Your reasoning may consist of many steps where you combine different numbers, after each combination, represent the current state between <state> and </state> tags.
Put your answer in between <answer> and </answer> tags in a format using == sign so it can be verified in python. 
If you do not follow this format you will be penalized heavily.

For example, if the 4 numbers are 3, 3, 5, 5, and your target was 30, you would go about this with

1. 3 * 5 = 15, so we can update state to <state> 15, 3, 5 </state>
2. 15 * 5 = 75, so we can update state to <state> 75, 3 </state>
3. This seems larger than our goal of 30, however we could combine with 5 with 3 to get <state> 15, 15 </state>
4. Ah and 15 + 15 = 30, so we can reach <state> 30 </state>

Every state generated should be one step away from a prior state, and ensure each number is used once.

<answer>
(3 * 5) + (3 * 5) == 30
</answer>
"""

MINI_SYS_PROMPT = '''
Game: Solve the 24 game with the given numbers.

Rules:
- Use each of the 4 numbers exactly once
- Use only +, -, *, / operations
- Show your step-by-step work
- After each step, show remaining numbers in <state></state> tags
- Put final expression in <answer></answer> tags using == format

Example with numbers 3, 3, 5, 5 and target 30:
1. 3 * 5 = 15 <state>15, 3, 5</state>
2. 3 * 5 = 15 <state>15, 15</state>
3. 15 + 15 = 30 <state>30</state>
<answer>(3 * 5) + (3 * 5) == 30</answer>
'''

XML_COT_FORMAT = """
<answer>
{answer}
</answer>
"""

import re

def extract_states(llm_response):
    """
    Extract all states from an LLM response as lists of integers.
    
    Args:
        llm_response (str): The full response from the LLM
        
    Returns:
        list: A list of lists, where each inner list contains the integers from a state
    """
    # Find all text between <state> and </state> tags
    state_pattern = r'<state>(.*?)</state>'
    state_matches = re.findall(state_pattern, llm_response, re.DOTALL)
    
    # Process each state string into a list of integers
    states = []
    for state_str in state_matches:
        # Split by commas or spaces, strip whitespace, and convert to numbers
        numbers = []
        # First split by commas if present
        if ',' in state_str:
            parts = state_str.split(',')
        else:
            # If no commas, split by spaces
            parts = state_str.split()
            
        for num_str in parts:
            # Clean and extract only numeric parts (including decimal points)
            cleaned = re.sub(r'[^\d.]', '', num_str.strip())
            if cleaned:
                try:
                    # Use float to preserve decimal values
                    num_val = float(cleaned)
                    # Convert to int only if it's a whole number
                    if num_val.is_integer():
                        numbers.append(int(num_val))
                    else:
                        numbers.append(num_val)
                except ValueError:
                    # Skip invalid numbers
                    pass
        states.append(numbers)
    
    return states


def is_valid_expression(expr):
    if expr == "":
        return -1
    try: 
        result = eval(expr)
        return 1 if result else 0
    except Exception as e:
        print(f'Error evaluating expression {str(e)}')
        return -1

def evaluate_expression(expression_string, inputs, goal):
    """
    Evaluates whether an expression is valid and true according to the following constraints:
    1. The left side of the expression uses all inputs exactly once
    2. The right side of the expression equals the goal
    
    Args:
        expression_string: String representing a mathematical expression with == operator
        inputs: List of numbers or values that must be used exactly once on the left side
        goal: The target value that the right side must equal
        
    Returns:
        boolean: True if the expression is valid and true, False otherwise
    """
    # Split the expression by the == sign
    if "==" not in expression_string:
        return False
    
    left_side, right_side = expression_string.split("==", 1)
    left_side = left_side.strip()
    right_side = right_side.strip()
    
    # Check if right side evaluates to the goal
    try:
        right_result = eval(right_side)
        if right_result != goal:
            return False
    except:
        return False
    
    # Check if left side uses all inputs exactly once
    try:
        left_result = eval(left_side)
        
        # Extract all numbers from the left side
        import re
        numbers_in_expression = [int(num) for num in re.findall(r'\d+', left_side)]
        
        # Sort both lists to compare
        sorted_inputs = sorted(inputs)
        sorted_numbers = sorted(numbers_in_expression)
        
        # Check if all inputs are used exactly once
        if sorted_inputs != sorted_numbers:
            return False
        
        # Check if the left side equals the right side
        return left_result == goal
    except:
        return False