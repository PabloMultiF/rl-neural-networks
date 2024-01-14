import numpy as np

np.random.seed(24)

class Environment:
    def __init__(self):
        self.initial_state = [0, 0] 
        self.current_state = [0, 0]
        self.final_state = [7, 5]
        self.action_penalty = -10 # Se penaliza cada accion para que se logre realizar la menor accion posible
        self.actions = {
            'arriba': [-1, 0],
            'abajo': [1, 0],
            'derecha': [0, 1],
            'izquerda': [0, -1]
        }
        self.rewards = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -100.0, 0.0, 0.0, 100.0, 0.0]
        ]
    
    def apply_action(self, action):
        previus_state = self.current_state
        next_state = [previus_state[0] + self.actions[action][0], previus_state[1] + self.actions[action][1]]

        if (next_state[0] > len(self.rewards)-1 or next_state[0] < 0):
            next_state = previus_state
        
        if (next_state[1] > len(self.rewards[next_state[0]])-1 or next_state[1] < 0):
            next_state = previus_state

        self.current_state = next_state

        reward = self.rewards[next_state[0]][next_state[1]] + self.action_penalty

        return reward
    
    def reset(self):
        self.current_state = [0, 0]

    def print_best_road(self, road):
        road_empty = []

        for fila in range(0, len(self.rewards)):
            road_empty.append([])
            for _ in range(0, len(self.rewards[fila])):
                road_empty[fila].append(' - ')
        
        for state in road:
            road_empty[state[0]][state[1]] = ' + '
            
        for fila in range(0, len(road_empty)):
            print(road_empty[fila])

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.ratio_exploration = 0.05
        self.learning_rate = 0.1
        self.discount_factor = 0.1
        self.q_table = [] # La Q-Table nos indica una proporcion de que tan bueno o malo es tomar una accion en determinado estado

        # Esto es para inicializar los valores de la Q-Table del agente
        for fila in range(0, len(self.environment.rewards)):
            self.q_table.append([])
            for columna in range(0, len(self.environment.rewards[fila])):
                self.q_table[fila].append([])
                for _ in range(0, len(list(self.environment.actions))):
                    self.q_table[fila][columna].append(0.0)

    def actualizar_q_table(self, old_state, new_state, reward, action_taken):
        # Para actualizar la Q-Table se debe realizar la siguiente ecuacion
        # Q'(s,a) = Q(s,a) + Delta * [R(s,a) + (Omega * max( Q(s',a') ) ) - Q(s,a)]
        # Q(s,a)    = Valor en la Q-Table del estado 'S' con la accion 'A'
        # R(s,a)    = Recompensa al realizar del estado 'S' haciendo la accion 'A'
        # Delta     = Es el "Factor de aprendizaje" (Learning Rate) que indica "cuanto" queremos aprender en cada acción.
        # Omega     = Es el 'Factor de descuento'. Se utiliza para penalizar el número de acciones tomadas.
        # Q(s',a')  = Valor en la Q-Table del nuevo estado 'S´' (Es nuevo porque ya se realizo la accion 'A') con la accion 'A´'
        
        idx_action_taken = list(self.environment.actions).index(action_taken) # Indice de la accion que tomamos

        actual_q_value_options = self.q_table[old_state[0]][old_state[1]] # Valores posibles guardados en la Q-Table del estado en el que estamos
        actual_q_value_taken = actual_q_value_options[idx_action_taken] # Valor guardado en la Q-Table de la accion que tomamos en el estado actual

        future_q_value_options = self.q_table[new_state[0]][new_state[1]] # Valores posibles guardados en la Q-Table en el nuevo estado en el que estaremos
        future_q_value_taken =  max(future_q_value_options)  # Valor guardado en la Q-Table de la accion con mayor recompensa que podemos obtener en el nuevo estado en el que estaremos

        self.q_table[old_state[0]][old_state[1]][idx_action_taken] = actual_q_value_taken + self.learning_rate * (reward + self.discount_factor * future_q_value_taken - actual_q_value_taken)
    
    def print_q_table(self):
        for fila in range(0, len(self.q_table)):
            for columna in range(0, len(self.q_table[fila])):
                print('Y:' + str(fila) + ', X:' + str(columna) + ' - ' + str(self.q_table[fila][columna]))

    def get_action(self, state):
        if np.random.uniform() < self.ratio_exploration:
            # Seleccionamos una opción al azar
            action = np.random.choice(list(self.environment.actions))
        else:
            # Seleccionamos la acción que nos de mayor valor. Si hay empate, Seleccionamos una al azar
            idx_action = np.random.choice(np.flatnonzero(
                self.q_table[state[0]][state[1]] == np.array(self.q_table[state[0]][state[1]]).max()
            ))
            action = list(self.environment.actions)[idx_action]

        return action

    def run(self, n_episodes):
        best_reward = None
        best_road = None

        for _ in range(0, n_episodes):
            is_final_state = False 
            total_reward = 0 # Esta es la recompensa total en este camino
            road = [self.environment.initial_state] # Esta es la lista con todos los estados en los cuales estubo el agente en su camino. El primer estado es el inicial, pero se van a ir acumulando con todos
            # los demas estados intermedios hasta el estado final
            
            while not is_final_state:
                old_state = self.environment.current_state # Obtenemos el estado viejo (antes de aplicar la accion)
                
                action = self.get_action(old_state) # Obtenemos la accion que nos de la mejor recompensa o aleatoriamente
                
                reward = self.environment.apply_action(action) # Aplicamos la accion al estado viejo
                
                new_state = self.environment.current_state # Obtenemos el estado nuevo (despues de aplicar la accion)

                total_reward += reward # Se suma la recompensa de esta accion con la total

                road.append(new_state) # Agrego el nuevo estado al camino, asi lo puedo guardar en caso de que tenga la mayor cantidad de recompensa
                
                self.actualizar_q_table(old_state, new_state, reward, action) # Una vez que hemos reunido todos los valores necesarios, actualizamos la Q-Table

                is_final_state = new_state == self.environment.final_state # Vemos si estamos en el estado final
                        
            if best_reward == None or total_reward > best_reward:
                best_reward = total_reward
                best_road =  road

            self.environment.reset()
        
        #print(best_award)
        self.environment.print_best_road(best_road)
        #agent.print_q_table()

    def run_once(self):
        road = []
        is_final_state = False
        while not is_final_state:
            old_state = self.environment.current_state # Obtenemos el estado viejo (antes de aplicar la accion)
                
            action = self.get_action(old_state) # Obtenemos la accion que nos de la mejor recompensa o aleatoriamente
                
            reward = self.environment.apply_action(action) # Aplicamos la accion al estado viejo
                
            new_state = self.environment.current_state # Obtenemos el estado nuevo (despues de aplicar la accion)

            is_final_state = new_state == self.environment.final_state # Vemos si estamos en el estado final
            
            road.append(new_state)

        print(road)

environment = Environment()
agent = Agent(environment)
agent.run(190)
agent.run_once()