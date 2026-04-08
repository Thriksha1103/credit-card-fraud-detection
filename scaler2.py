import random

class TreasureHuntEnv:
    def __init__(self, size=5, obstacles=0):
        self.size = size
        self.obstacles_count = obstacles
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0

        self.obstacles = set()
        while len(self.obstacles) < self.obstacles_count:
            pos = (random.randint(0, self.size - 1),
                   random.randint(0, self.size - 1))
            if pos != (0, 0) and pos != tuple(self.goal_pos):
                self.obstacles.add(pos)

        return self.state()

    def state(self):
        return {
            "agent": tuple(self.agent_pos),
            "goal": tuple(self.goal_pos),
            "obstacles": list(self.obstacles)
        }

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos

        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        self.agent_pos = [x, y]

        reward = -1
        done = False

        if (x, y) in self.obstacles:
            return self.state(), -10, True
        if [x, y] == self.goal_pos:
            return self.state(), 100, True

        return self.state(), reward, done