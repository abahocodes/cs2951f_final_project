class Transition:
    def __init__(self, current_state, action, goal, reward, next_state, satisfied_goals_t):
        self.current_state = current_state
        self.action = action
        self.goal = goal
        self.reward = reward
        self.next_state = next_state
        self.satisfied_goals_t = satisfied_goals_t
    

