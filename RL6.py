import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
        self.current_state = self.initial_state

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.discount_factor = 0.1
        self.max_memory = 100
        self.explotation_rate = 0.95
        self.max_steps = 500
        self.max_explotation_rate = self.explotation_rate
        self.memory = []
        self.model = self.create_model()

    def get_action(self, state):
        if np.random.randn() > self.explotation_rate: # En caso de que sea menor que el ratio de exploracion, el algoritmo elijira un camino aleatorio
            return np.random.choice(list(self.environment.actions.keys()))
        else: # En caso de que sea mayor o igual que el ratio de exploracion
            result = self.model.predict([state], verbose=False)
            idx_action = np.argmax(result)
            return list(self.environment.actions)[idx_action]
    
    def get_action_safe(self, state):
        result = self.model.predict([state], verbose=False)
        idx_action = np.argmax(result)
        return list(self.environment.actions)[idx_action]
                
    def create_model(self):
        input_dim = len(self.environment.initial_state)
        output_dim = len(self.environment.actions)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        return model

    def print_road(self, road):
        for fila in range(0, len(self.environment.rewards)):
            fila_display = ""
            for columna in range(0, len(self.environment.rewards[fila])):
                try:
                    road.index([fila, columna]) # Si existe no disparara ningun error
                    fila_display = fila_display + ' + '
                except:
                    fila_display = fila_display + ' - '

            print(fila_display)

    def remember(self, old_state, new_state, reward, action):
        self.memory.append((old_state, new_state, reward, action))
        
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def learn(self, num_episode):
        # Se obtiene 100 elementos aleatorios de la memoria si esta tiene mas de 100 elementos
        # De lo contrario se toma TODOS los elementos de la memoria
        # Esto se hace para que el modelo no se demore mucho en aprender
        batch = (random.sample(self.memory, 100) if len(self.memory) > 100 else random.sample(self.memory, len(self.memory)))

        for old_state, new_state, reward, action in batch:
            q_values = self.model.predict([old_state], verbose=0) # El modelo predice dado el estado en el que esta actualmente, a donde deberia ir
            
            idx_action = list(environment.actions).index(action) # Obtiene el indice de la accion que se ha tomado en este estado

            # Devuelve un arreglo con 4 elementos donde cada elemento es la accion que deberia 
            # hacerse en ese estado y su valor es la accion 'probabilidad' de que dicha accion sea correcta
            # Un ejemplo de salida podria ser: [[-0.02234588  0.01137727 -0.02956272 -0.0098047 ]]
            q_values_future = self.model.predict([new_state], verbose=0)
            
            # Ahora obtenemos el elemento de mayor valor
            # Siguiendo el ejemplo anterior obtendriamos lo siguiente: 0.01137727
            q_values_future_max = np.amax(q_values_future[0])
            
            # Se le agrega a la accion que se predijo en el viejo estado la recompensa
            q_values[0][idx_action] = (reward + self.discount_factor * q_values_future_max) 

            # Ahora entreno a la red neuronal para que en el caso de que reciba el estado actual, 
            # me devuelva el valor con la recompensa o algo muy parecido
            self.model.fit(np.array([old_state]), q_values, epochs=1, verbose=0)

        self.explotation_rate = self.max_explotation_rate - (self.max_explotation_rate / (num_episode + 1)) # Actualizo el ratio de explotación

    def run_once(self):
        road = [self.environment.initial_state]
        
        final_state = False

        while not final_state:
            old_state = self.environment.current_state

            action = self.get_action_safe(old_state)

            environment.apply_action(action)

            new_state = self.environment.current_state

            final_state = new_state == self.environment.final_state

            road.append(new_state)

        print(road)

        self.print_road(road)

    def run(self, n_episodes = 10):

        for num_episode in range(0, n_episodes):
            print('Entrenando el episodio ' + str(num_episode))
            
            final_state = False

            num_steps = 0
            
            while not final_state:
                old_state = self.environment.current_state # Obtiene el estado actual del agente

                action = self.get_action(old_state) # Obtiene la accion que devuelva la mayor recompensa posible o una accion aleatoria

                environment.apply_action(action) # Aplica la accion al estado actual

                new_state = self.environment.current_state # Obtiene el estado actual del agente despues de haber aplicado la accion

                reward = self.environment.get_reward_state(new_state) # Obtiene la recompensa en el estado actual

                final_state = new_state == self.environment.final_state # Obtiene si el nuevo estado es el final del recorrido

                self.remember(old_state, new_state, reward, action) # Guarda el estado actual en la memoria para poder aprender de el cuando el recorrido halla terminado

                if final_state or num_steps > self.max_steps:
                    self.learn(num_episode)
                    self.memory = []
                    num_steps = 0
                
                num_steps += 1
                #print(num_steps)
                
            self.environment.reset_state() # Devolvemos el estado del agente al inicio
        
environment = Environment()
agent = Agent(environment)
agent.run(45)
agent.run_once()

