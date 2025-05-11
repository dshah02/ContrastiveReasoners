

def get_reward_func(eval_func):
    def reward_func(completions, item, **kwargs):
        reward = []
        correct = []
        print(f"Input numbers: {item[0]['nums']}")
        print(f"Target: {item[0]['target']}")
        print("-" * 40)
        for pred, it in zip(completions, item):
            res, pred_sol = eval_func(pred, data_item = it)
            acc, partial_acc, extraction_rate = res["accuracy"], res["partial_accuracy"], res["extraction_rate"]
            correct.append(acc)
        
            print(f"Parsed output: {pred_sol['parsed_output']}")
            print("-" * 40)
            reward.append(4 * acc)
        # print(completions[0])
        print(f"Correct: {sum(correct)}/{len(correct)} = {sum(correct)/len(correct):.2%}")
        return reward
    return reward_func

def get_partial_acc_func(eval_func):
    def partial_acc_func(completions, item, **kwargs):
        reward = []
        correct = []
        for pred, it in zip(completions, item):
            res, pred_sol = eval_func(pred, data_item = it)
            acc, partial_acc, extraction_rate = res["accuracy"], res["partial_accuracy"], res["extraction_rate"]
            reward.append(2 * partial_acc)
        return reward
    return partial_acc_func

def get_extraction_rew_func(eval_func):
    def extraction_rew(completions, item, **kwargs):
        reward = []
        correct = []
        for pred, it in zip(completions, item):
            res, pred_sol = eval_func(pred, data_item = it)
            acc, partial_acc, extraction_rate = res["accuracy"], res["partial_accuracy"], res["extraction_rate"]
            reward.append(extraction_rate)
        return reward
    return extraction_rew
