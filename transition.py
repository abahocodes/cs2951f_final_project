class Transition:
    def __init__(self, current_state, action, goal, reward, next_state, satisfied_goals_t, done):
        self.current_state = current_state
        self.action = action
        self.goal = goal
        self.reward = reward
        self.next_state = next_state
        self.satisfied_goals_t = satisfied_goals_t
        self.done = done
        
    def __str__(self):
        p = ''
        p += 'Current State:\t' + str(self.current_state) + '\n'
        p += 'Action :\t' + str(self.action) + '\n'
        p += 'Goal :\t' + str(self.goal) + '\n'
        p += 'Next State: ' + str(self.next_state)
        return p
    

