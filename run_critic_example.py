import torch
from critics.critic9 import ContrastiveCritic9

model_path = './critic_checkpoints/critic8_model_epoch_176.pt'

critic = ContrastiveCritic9()
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    critic.load_state_dict(state_dict)
    print(f"loaded model from {model_path}")
except Exception as e:
    print(f"Error: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
critic.to(device)

states = [
    "",
    "",
    "",
]
actions = [
    "|- Pick two numbers (21, 16) (numbers left: [17, 26]). Try possible operations. ### Current State: [21, 16, 17, 26]",
    "|- Try 37 - 17 = 20. Add 20 to the number set. Current number set: [20, 26], target: 46, just two numbers left. ### Current State: [20, 26]",
    "|- Try 26 + 20 = 46. Evaluate 46 == 46, target found! ### Current State: [46]"
]
goals = [
    "[20]",
    "[46]",
    "[46]"
]

with torch.no_grad():
    Q = critic.forward(states, actions, goals)
    print("Q values (rows: (state, action), columns: goal):")
    print(Q.cpu().numpy())
    print()
    for i, (s, a) in enumerate(zip(states, actions)):
        for j, g in enumerate(goals):
            print(f"Q(({i}) {s} + {a}, ({j}) {g}): {Q[i, j].item():.4f}")