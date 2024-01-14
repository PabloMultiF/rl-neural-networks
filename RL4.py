import numpy as np
#np.random.seed(20)

class Environment:
    def __init__(self):
        self.initial_state = [0, 0]
        self.current_state = [0, 0]
        self.final_state = [4, 4]
        self.action_penalty = -5.0
        self.rewards = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 100.0],
        ]
        self.actions = {
            'arriba': [-1, 0],
            'abajo': [1, 0],
            'derecha': [0, 1],
            'izquierda': [0, -1]
        }

    def apply_action(self, action):
        previus_state = self.current_state
        new_state = [self.current_state[0]+self.actions[action][0], self.current_state[1]+self.actions[action][1]]

        if (new_state[0] > len(self.rewards)-1 or new_state[0] < 0):
            new_state = previus_state

        if (new_state[1] > len(self.rewards[0])-1 or new_state[1] < 0):
            new_state = previus_state

        self.current_state = new_state

    def get_reward_state(self, state):
        return self.rewards[state[0]][state[1]] + self.action_penalty
    
    def reset_state(self):
        self.current_state = [0, 0]

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.discount_factor = 0.1
        self.learning_rate = 0.1
        self.exploration_ratio = 0.00005
        self.q_table = []
        self.best_road = []
        for fila in range(0, len(self.environment.rewards)):
            self.q_table.append([])
            for columna in range(0, len(self.environment.rewards[fila])):
                self.q_table[fila].append([])
                for _ in range(0, len(self.environment.actions)):
                    self.q_table[fila][columna].append(0.0)

    def get_action(self, state, verbose = False):
        if np.random.randn() < self.exploration_ratio: # En caso de que sea menor que el ratio de exploracion, el algoritmo elijira un camino aleatorio
            if verbose:
                print('He tomado la opcion aleatoria')
            return np.random.choice(list(self.environment.actions.keys()))
        else: # En caso de que sea mayor o igual que el ratio de exploracion
            if verbose:
                print('He tomado la opcion seguro')
            idx_action = np.random.choice(np.flatnonzero(
                self.q_table[state[0]][state[1]] == np.array(self.q_table[state[0]][state[1]]).max()
            ))
            next_action = list(self.environment.actions)[idx_action]
            
            return next_action

    def print_best_road(self):
        print(self.best_road)

    def print_road(self):
        for fila in range(0, len(self.environment.rewards)):
            fila_display = ""
            for columna in range(0, len(self.environment.rewards[fila])):
                try:
                    self.best_road.index([fila, columna]) # Si existe no disparara ningun error
                    fila_display = fila_display + ' + '
                except:
                    fila_display = fila_display + ' - '

            print(fila_display)
        

    def print_q_table(self):
        for fila in range(0, len(self.q_table)):
            for columna in range(0, len(self.q_table[fila])):
                print('(Y:' + str(fila) + 'X:' + str(columna) + ') ' + str(self.q_table[fila][columna]))
    
    def update_q_table(self, old_state, new_state, reward, action_prima, action):
        index_action = list(self.environment.actions).index(action)
        
        actual_state_q_table_value = self.q_table[old_state[0]][old_state[1]][index_action] # Q(s, a)

        index_action_prima = list(self.environment.actions).index(action_prima)
        
        future_state_q_table_value = self.q_table[new_state[0]][new_state[1]][index_action_prima] # Q(s', a')

        q_function_value = actual_state_q_table_value + self.learning_rate * (reward + self.discount_factor * future_state_q_table_value - actual_state_q_table_value)

        self.q_table[old_state[0]][old_state[1]][index_action] = q_function_value

    def run_once(self):
        road = []
        
        final_state = False

        while not final_state:
            old_state = self.environment.current_state

            action = self.get_action(old_state, True)

            environment.apply_action(action)

            new_state = self.environment.current_state

            final_state = new_state == self.environment.final_state

            road.append(new_state)

        print(road)

    def run(self, n_episodes = 10):
        best_reward = None

        for _ in range(0, n_episodes):
            final_state = False

            total_reward = 0

            road = [self.environment.initial_state]
            
            while not final_state: # Algoritmo de SARSA LEARNING
                old_state = self.environment.current_state

                action = self.get_action(old_state)

                environment.apply_action(action)

                new_state = self.environment.current_state

                reward = self.environment.get_reward_state(new_state)

                road.append(new_state)

                action_prima = self.get_action(new_state)

                environment.apply_action(action_prima)

                new_state_prima = self.environment.current_state

                self.update_q_table(old_state, new_state_prima, reward, action_prima, action)

                final_state = new_state_prima == self.environment.final_state

                total_reward += reward

                road.append(new_state_prima)

            if best_reward == None or best_reward < total_reward:
                self.best_road = road
                best_reward = total_reward
                        
            self.environment.reset_state()
        
environment = Environment()
agent = Agent(environment)
agent.run(400)
#print(agent.q_table)
agent.print_road()
#agent.print_best_road()
agent.run_once()

