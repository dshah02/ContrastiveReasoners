import unittest
import re
import numpy as np
from strings import extract_states, is_valid_expression, evaluate_expression
from verify_states import get_next_states, list_all_valid_states, is_valid_state

class TestGameFunctions(unittest.TestCase):
    
    def test_extract_states(self):
        # Test with commas
        response1 = "Here is step 1 <state>15, 3, 5</state> and step 2 <state>15, 15</state>"
        expected1 = [[15, 3, 5], [15, 15]]
        self.assertEqual(extract_states(response1), expected1)
        
        # Test with spaces
        response2 = "Step 1 <state>4 7 2</state> next step <state>11 2</state>"
        expected2 = [[4, 7, 2], [11, 2]]
        self.assertEqual(extract_states(response2), expected2)
        
        # Test with decimal numbers
        response3 = "Step 1 <state>4.5, 7, 2</state> next is <state>11.5, 2</state>"
        expected3 = [[4.5, 7, 2], [11.5, 2]]
        self.assertEqual(extract_states(response3), expected3)
        
        # Test with messy formatting
        response4 = "Messy <state> 10 , 5,3 </state> and <state>15,   3   </state>"
        expected4 = [[10, 5, 3], [15, 3]]
        self.assertEqual(extract_states(response4), expected4)
        
        # Test with no states
        response5 = "There are no states here"
        expected5 = []
        self.assertEqual(extract_states(response5), expected5)
        
        # Test with empty state
        response6 = "Empty state <state></state>"
        expected6 = [[]]
        self.assertEqual(extract_states(response6), expected6)

    def test_is_valid_expression(self):
        # Test valid expressions
        self.assertEqual(is_valid_expression("2 + 2"), 1)
        self.assertEqual(is_valid_expression("10 - 5"), 1)
        self.assertEqual(is_valid_expression("0"), 0)  # Evaluates to False
        
        # Test invalid expressions
        self.assertEqual(is_valid_expression("2 +"), -1)
        self.assertEqual(is_valid_expression("/ 4"), -1)
        self.assertEqual(is_valid_expression("x + y"), -1)

    def test_evaluate_expression(self):
        # Test valid expressions that use all inputs
        self.assertTrue(evaluate_expression("2 + 3 + 4 + 5 == 14", [2, 3, 4, 5], 14))
        self.assertTrue(evaluate_expression("(3 * 4) + 2 - 5 == 9", [3, 4, 2, 5], 9))
        self.assertTrue(evaluate_expression("8 / 2 + 3 * 4 == 16", [8, 2, 3, 4], 16))
        
        # Test valid expression but wrong result
        self.assertFalse(evaluate_expression("2 + 3 + 4 + 5 == 15", [2, 3, 4, 5], 14))
        
        # Test using wrong inputs
        self.assertFalse(evaluate_expression("2 + 3 + 4 + 6 == 15", [2, 3, 4, 5], 15))
        
        # Test missing inputs
        self.assertFalse(evaluate_expression("2 + 3 + 4 == 9", [2, 3, 4, 5], 9))
        
        # Test duplicate inputs
        self.assertFalse(evaluate_expression("2 + 2 + 3 + 5 == 12", [2, 3, 4, 5], 12))
        
        # Test malformed expressions
        self.assertFalse(evaluate_expression("2 + 3 + 4 + 5", [2, 3, 4, 5], 14))
        self.assertFalse(evaluate_expression("== 14", [2, 3, 4, 5], 14))
        self.assertFalse(evaluate_expression("2 + 3 + 4 + 5 == ", [2, 3, 4, 5], 14))

    def test_full_game_example(self):
        """Test a complete game sequence with the example from the prompt"""
        sample_response = """
        1. 3 * 5 = 15, so we can update state to <state> 15, 3, 5 </state>
        2. 15 * 5 = 75, so we can update state to <state> 75, 3 </state>
        3. This seems larger than our goal of 30, however we could combine with 5 with 3 to get <state> 15, 15 </state>
        4. Ah and 15 + 15 = 30, so we can reach <state> 30 </state>

        <answer>
        (3 * 5) + (3 * 5) == 30
        </answer>
        """
        
        # Check that states are correctly extracted
        expected_states = [[15, 3, 5], [75, 3], [15, 15], [30]]
        self.assertEqual(extract_states(sample_response), expected_states)
        
        # Extract the answer
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, sample_response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        # Verify the answer
        self.assertTrue(evaluate_expression(answer, [3, 3, 5, 5], 30))

    def test_get_next_states(self):
        """Test the get_next_states function with various inputs"""
        # Test with 4 numbers
        state1 = [1, 2, 3, 4]
        next_states1 = get_next_states(state1)
        # Should have 36 possible next states (6 pairs * 6 operations)
        # Some operations might be skipped due to division by zero checks
        self.assertEqual(len(next_states1), 36)
        
        # Test with specific examples
        # For [1,2] remaining [3,4], operations: 1+2=3 -> [3,3,4]
        self.assertTrue(any(np.array_equal(state, [3, 3, 4]) for state in next_states1))
        # For [3,4] remaining [1,2], operations: 3+4=7 -> [1,2,7]
        self.assertTrue(any(np.array_equal(state, [1, 2, 7]) for state in next_states1))
        
        # Test with 3 numbers
        state2 = [3, 3, 4]
        next_states2 = get_next_states(state2)
        # Should have 18 possible next states (3 pairs * 6 operations)
        self.assertEqual(len(next_states2), 18)
        # Test specific results: [3,3] + [4] -> [3,7], [3,4] + [3] -> [3,7]
        self.assertTrue(any(np.array_equal(state, [3, 7]) for state in next_states2))
        
        # Test with 2 numbers
        state3 = [6, 4]
        next_states3 = get_next_states(state3)
        # Should have 6 possible next states (1 pair * 6 operations)
        self.assertEqual(len(next_states3), 6)
        # Check specific results: 6+4=10 -> [10], 6*4=24 -> [24]
        self.assertTrue(any(np.array_equal(state, [10]) for state in next_states3))
        self.assertTrue(any(np.array_equal(state, [24]) for state in next_states3))
        
        # Test with 1 number - should return empty list
        state4 = [24]
        next_states4 = get_next_states(state4)
        self.assertEqual(next_states4, [])
        
        # Test with empty state - should return empty list
        state5 = []
        next_states5 = get_next_states(state5)
        self.assertEqual(next_states5, [])
        
        # Test with division by zero
        state6 = [0, 5]
        next_states6 = get_next_states(state6)
        # Should have 5 possible next states (1 pair * 5 operations)
        # Division 5/0 should be skipped
        self.assertEqual(len(next_states6), 5)
        # 0+5=5, 0-5=-5, 5-0=5, 0*5=0, 0/5=0
        self.assertTrue(any(np.array_equal(state, [5]) for state in next_states6))
        self.assertTrue(any(np.array_equal(state, [-5]) for state in next_states6))
        self.assertTrue(any(np.array_equal(state, [0]) for state in next_states6))

    def test_list_all_valid_states(self):
        """Test the list_all_valid_states function"""
        # Test with one state
        states1 = [[1, 2, 3, 4]]
        all_states1 = list_all_valid_states(states1)
        print(all_states1)
        self.assertEqual(len(all_states1), 37)  # Same as get_next_states for this input
        
        # Test with multiple states
        states2 = [[1, 2, 3], [4, 5, 6]]
        all_states2 = list_all_valid_states(states2)
        # Each state should generate 18 next states (3 pairs * 6 operations)
        self.assertEqual(len(all_states2), 38)  # 18 + 18
        
        # Test with empty list
        states3 = []
        all_states3 = list_all_valid_states(states3)
        self.assertEqual(all_states3, [])
        
        # Test with states that can't generate next states
        states4 = [[1], []]
        all_states4 = list_all_valid_states(states4)
        self.assertEqual(all_states4, [[1], []])

    def test_is_valid_state(self):
        """Test the is_valid_state function"""
        # Valid state generation
        self.assertTrue(is_valid_state([1, 3], [[1, 2, 3]]))  # 1+2=3 -> [1,3]
        self.assertTrue(is_valid_state([0, 4], [[2, 2, 4]]))  # 2-2=0 -> [0,4]
        self.assertTrue(is_valid_state([8], [[2, 4]]))        # 2*4=8 -> [8]
        
        # Invalid state generation
        self.assertFalse(is_valid_state([5, 7], [[1, 2, 3]]))  # Can't generate [5,7] from [1,2,3]
        self.assertFalse(is_valid_state([10], [[1, 2, 3]]))    # Can't generate [10] from [1,2,3] in one step
        
        # Test with almost equal values (floating point comparison)
        self.assertTrue(is_valid_state([2.0001], [[1, 2]]))    # 1+1=2 -> [2] (within tolerance)
        
        # Test with multiple states
        self.assertTrue(is_valid_state([3], [[1, 2], [3, 4]]))  # Valid based on [1,2]
        
        # Test with edge cases
        self.assertFalse(is_valid_state([], [[1, 2, 3]]))       # Empty state not valid
        self.assertFalse(is_valid_state([1, 2, 3], [[1, 2]]))   # Can't generate more numbers


if __name__ == "__main__":
    unittest.main()