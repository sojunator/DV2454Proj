from DQNAgent import DQNAgent
from pathlib import Path

my_file = Path("my_model.hd5")
if my_file.is_file():
    Agent = DQNAgent("my_model.hd5")
else:
    Agent = DQNAgent()


for i in range(1500):
    Agent.observe()
    Agent.train()

rewards = 0
for _ in range(10):
    rewards += Agent.play()

print(rewards / 3)
Agent.save_network("my_model.hd5")
