SYSTEM_PROMPT = """
We will be playing the game 24. In this game you will start with 4 numbers from 1 to 10 and aim to generate a target number.
To generate the target number, you will combine the 4 starting numbers using +, -, *, and /. 
Your reasoning may consist of many steps where you combine different numbers, after each combination, represent the current state between <state> and </state> tags.
Put your answer in between <answer> and </answer> tags in a format using == sign so it can be verified in python. 
If you do not follow this format you will be penalized heavily.

For example, if the 4 numbers are 3, 3, 5, 5, and your target was 30, you would go about this with

1. 3 * 5 = 15, so we can update state to <state> 15, 3, 5 </state>
2. 15 * 5 = 75, so we can update state to <state> 75, 3 </state>
3. This seems larger than our goal of 30, however we could combine with 5 with 3 to get <state> 15, 15 </state>
4. Ah and 15 + 15 = 30, so we can reach <state> 30 </state>

Every state generated should be one step away from a prior state.

<answer>
(3 * 5) + (3 * 5) == 30
</answer>
"""

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
        # Split by commas, strip whitespace, and convert to integers
        numbers = []
        for num_str in state_str.split(','):
            # Clean and extract only numeric parts
            cleaned = re.sub(r'[^\d.]', '', num_str.strip())
            if cleaned:
                numbers.append(int(float(cleaned)))
        states.append(numbers)
    
    return states


def evaluate_expression(expr):
    try: 
        result = eval(expr)
        return 1 if result else 0
    except Exception as e:
        print(f'Error evaluating expression {str(e)}')
        return -1
