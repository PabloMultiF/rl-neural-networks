# importamos las librerias necesarias
import pandas as pd
import numpy as np
import itertools
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Formato de los decimales en Pandas y la semilla del Random
pd.options.display.float_format = '{:,.2f}'.format
np.random.seed(5)


class Environment(object):
    def __init__(self, action_penalty=-1.0):
        """
        Clase que representa y controla en entorno
        :param action_penalty:    Factor de descuento del Reward por acción tomada
        """
        self.actions = {'Arriba': [-1, 0],
                        'Abajo': [1, 0],
                        'Izquierda': [0, -1],
                        'Derecha': [0, 1]}
        self.rewards = [[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -100.0, 100.0]]
        self.action_penalty = action_penalty         # Penalización por cada paso dado
        self.state = [0, 0]                          # Estado en el que se encuentra el agente
        self.final_state = [3, 3]                    # Estado final del entorno. Cuando el agente llega, se termina el episodio
        self.total_reward = 0.0                      # Contador de recompensas en el episodio
        self.actions_done = []                       # Lista en la que se guardan los pasos (acciones) realizadas en cada episodio

    def reset(self):
        """
        Método que reinicia las variables del entorno y devuelve es estado inicial
        :return:    state
        """
        self.total_reward = 0.0    # Inicializamos Reward a 0
        self.state = [0, 0]        # Posicionamos al agente en el estado inicial
        self.actions_done = []     # Inicializamos la listas de pasos (acciones)
        return self.state

    def step(self, action):
        """
        Método que ejecuta una acción determinada del conjunto de acciones {Arriba, Abajo, Izquierda, Derecha}
        para guiar al agente en el entorno.
        :param action:    Acción a ejecutar
        :return:          (state, reward, is_final_state)
        """
        self.apply_action(action)                                                  # Realizamos la acción (cambio de estado)
        self.actions_done.append(self.state[:])                                    # Guardamos el paso (accion) realizada
        is_final_state = np.array_equal(self.state, self.final_state)              # Comprobamos si hemos llegado al estado final
        reward = self.rewards[self.state[0]][self.state[1]] + self.action_penalty  # Calculamos el reward (recompensa) por la acción tomada
        self.total_reward += reward                                                # Sumamos el reward (recompensa) total del episodio
        return self.state, reward, is_final_state                                  # Devolvemos es estado, el reward (recompensa) y si hemos llegado al estado final

    def apply_action(self, action):
        """
        Método que calcula el nuevo estado a partir de la acción a ejecutar
        :param action:    Acción a ejecutar
        """
        self.state[0] += self.actions[action][0]
        self.state[1] += self.actions[action][1]

        # Si nos salimos del tablero por arriba o por abajo, nos quedamos en la posicion que estabamos
        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] > len(self.rewards) - 1:
            self.state[0] -= 1

        # Si nos salimos del tablero por los lados, nos quedamos en la posicion que estabamos
        if self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] > len(self.rewards[0]) - 1:
            self.state[1] -= 1

    def print_path_episode(self):
        """
        Método que imprime por pantalla el camino seguido por el agente
        :return:
        """
        path = [['-' for _ in range(len(self.rewards))] for _ in range(len(self.rewards[0]))]
        path[0][0] = '0'
        for index, step in enumerate(self.actions_done):
            path[step[0]][step[1]] = str(index + 1)

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in path]),
                           index=["x{}".format(str(i)) for i in range(len(path))],
                           columns=["y{}".format(str(i)) for i in range(len(path[0]))]))

class DeepQLearner(object):

    def __init__(self, environment, max_memory=100, discount_factor=0.1, explotation_rate=0.95, max_steps=500):
        """
        Clase que implementa el Algoritmo de Aprendizaje Deep Q-Learning
        :param environment:         Entorno en el que tomar las acciones
        :param max_memory:          Número maximo de acciones a memorizar (guardar) en un episodio
        :param discount_factor:     Factor de descuento (0=Estrategia a corto plazo, 1=Estrategia a largo plazo)
        :param explotation_rate:    Ratio de explotación
        :param max_steps:           Número máximo de pasos a ejecutar en un episodio
        """
        self.environment = environment
        self.memory = list()               # Estado Actual (S_t), Acción realizada (a_t), Reward (R(s,t)), Estado Siguiente (S_t+1), ¿Estado Final?
        self.max_memory = max_memory
        self.model = self.create_model()   # Red Neuronal
        self.discount_factor = discount_factor
        self.max_explotation_rate = explotation_rate
        self.explotation_rate = 0
        self.max_steps = max_steps

    @property
    def name(self):
        return 'Deep Q-Learner'

    def create_model(self):
        """
        Función que crea y devuelve la red neuronal
        :return: Red Neuronal
        """

        input_dim = len(self.environment.state)      # Número de neuronas de la capa de entrada '2' (X,Y)
        output_dim = len(self.environment.actions)   # Número de neuronas de la capa de salida '4' (estados)

        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        return model

    def get_next_action(self, state):
        """
        Método que selecciona la siguiente acción a tomar:
            Aleatoria ->     si el ratio de explotación es inferior al umbral
            Mejor Acción ->  si el ratio de explotación es superior al umbral
        :param state:   Estado del agente
        :return:        next_action
        """

        if np.random.uniform() > self.explotation_rate:
            # Seleccionamos una acción al azar
            next_action = np.random.choice(list(self.environment.actions))
        else:
            # Seleccionamos la acción que nos de mayor valor.
            qus = self.model.predict([state])
            idx_action = np.argmax(qus[0])
            next_action = list(self.environment.actions)[idx_action]

        return next_action

    def update(self, environment, state, action, reward, new_state, is_final_state, num_episode, num_steps):
        """
        Método que implementa el Algoritmo de Aprendizaje Deep Q-Learning
        :param environment:       Entorno en el que tomar las acciones
        :param state:             Estado actual
        :param action:            Acción a realizar
        :param reward:            Recompensa obtenida por la acción tomada
        :param new_state:         Nuevo estado al que se mueve el agente
        :param is_final_state:    Boolean. Devuelve True si el agente llega al estado final; si no, False
        :param num_episode:       Número de episodios ejecutados
        :param num_steps:         Número de pasos dados en el episodio
        """
        self.remenber(state=state, action=action, reward=reward, new_state=new_state, is_final_state=is_final_state)
        if is_final_state or num_steps > self.max_steps:
            self.learn(environment=environment, num_episode=num_episode)
            self.reset()

    def remenber(self, state, action, reward, new_state, is_final_state):
        """
        Método que guarda en una lista, una tupla con información de cada uno de los pasos
        realizados por el agente en el entorno durante el episodio. En el caso de que el numero
        de acciones en la memoria sea superior al número de acciones máximas a guardar, iremos
        eliminando la acción más antigua de la lista.
        :param state:             Estado actual
        :param action:            Acción a realizar
        :param reward:            Recompensa obtenida por la acción tomada
        :param new_state:         Nuevo estado al que se mueve el agente
        :param is_final_state:    Boolean. Devuelve True si el agente llega al estado final; si no, False
        """

        self.memory.append((state, action, reward, new_state, is_final_state))
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def learn(self, environment, num_episode):
        """
        Método que actualiza el modelo (Red Neuronal) - Aprende de las acciones realizadas en el episodio.
        Este método también actualiza el ratio de explotación de las siguiente manera:
        ration_explotacion = ratio_explotación - (maximo_ratio_explotacion / (num_episodios + 1))
        :param environment:       Entorno en el que tomar las acciones
        :param num_episode:       Número del episodio
        """
        batch = (random.sample(self.memory, 100)
                 if len(self.memory) > 100 else random.sample(self.memory, len(self.memory)))

        for state, action, reward, new_state, is_final_state in batch:
            q_values = self.model.predict([state])
            idx_action = list(environment.actions).index(action)

            q_values[0][idx_action] = (reward + (self.discount_factor * np.amax(self.model.predict([new_state])[0]))
                                       if not is_final_state else reward)

            self.model.fit(np.array([state]), q_values, epochs=1, verbose=0)

        # Actualizo el ratio de explotación
        self.explotation_rate = self.max_explotation_rate - (self.max_explotation_rate / (num_episode + 1))
        print('El nuevo ratio de explotacion es ' + str(self.explotation_rate))

    def reset(self):
        """
        Método que vacia la lista (la memoria) con los pasos realizados por el agente
        :return: 
        """
        del self.memory[:]

    def print_q_table(self):
        """
        Método que imprime por pantalla la Q-Table aprendida por la red
        """
        states = list(itertools.product([0, 1, 2, 3], repeat=2))  # Generamos todos los posibles estados
        q_table = self.model.predict(states)                      # Predecimos con la red los Q(s,a)
        df = (pd.DataFrame(data=q_table,                          # Pasamos la Q_Table a un DataFrame
                           columns=['Arriba', 'Abajo', 'Izquierda', 'Derecha']))
        df.insert(0, 'Estado', ['x{},y{}'.format(state[0], state[1]) for state in states])
        print(df.to_string(index=False))

    def print_best_actions_states(self):
        """
        Método que imprime por pantalla el valor de la mejor opción a realizar en cada uno de los estados
        """
        states = list(itertools.product([0, 1, 2, 3], repeat=2))  # Generamos todos los posibles estados
        q_table = self.model.predict(states)  # Predecimos con la red los Q(s,a)

        best = (np.array([list(self.environment.actions)[np.argmax(row)] for row in q_table])
                .reshape(len(self.environment.rewards), len(self.environment.rewards[0])))

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]),
                           index=["x{}".format(str(i)) for i in range(len(best))],
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))]))

    def print_best_values_states(self):
        """
        Método que imprime por pantalla el valor de la mejor opción a realizar en cada uno de los estados
        """
        states = list(itertools.product([0, 1, 2, 3], repeat=2))  # Generamos todos los posibles estados
        q_table = self.model.predict(states)                      # Predecimos con la red los Q(s,a)

        best = (np.array([[np.max(row) for row in q_table]])
                .reshape(len(self.environment.rewards), len(self.environment.rewards[0])))

        print(pd.DataFrame(data=np.array([np.array(xi) for xi in best]),
                           index=["x{}".format(str(i)) for i in range(len(best))],
                           columns=["y{}".format(str(i)) for i in range(len(best[0]))]))

from copy import deepcopy


def run_agent(learner=DeepQLearner, num_episodes=10, learning_rate=0.1, discount_factor=0.1, ratio_explotacion=0.95,
              max_steps=500, verbose=False):
    """
    Método que ejecuta el proceso de aprendizaje del agente en un entorno
    :param learner:              Algoritmo de Aprendizaje
    :param num_episodes:         Número de veces que se ejecuta (o aprende) el agente en el entorno
    :param learning_rate:        Factor de Aprendizaje
    :param discount_factor:      Factor de descuento (0=Estrategia a corto plazo, 1=Estrategia a largo plazo)
    :param ratio_explotacion:    Ratio de explotación
    :param max_steps:            TODO
    :param verbose:              Boolean, si queremos o no imprimir por pantalla información del proceso
    """

    # Instanciamos el entorno
    environment = Environment()

    # Instanciamos el método de aprendizaje
    learner = learner(environment=environment,
                      max_memory=100,
                      discount_factor=discount_factor,
                      explotation_rate=ratio_explotacion,
                      max_steps=max_steps)

    last_episode = None

    for n_episode in range(0, num_episodes):
        state = environment.reset()
        is_final_state = False
        num_steps_episode = 0
        while not is_final_state:
            old_state = state[:]
            next_action = learner.get_next_action(state=old_state)             # Accion a realizar; explotando o explorando
            new_state, reward, is_final_state = environment.step(next_action)  # Realizamos la accion

            learner.update(environment=environment,  # Actualizamos el entorno
                           state=deepcopy(old_state),
                           action=next_action,
                           reward=reward,
                           new_state=deepcopy(new_state),
                           is_final_state=is_final_state,
                           num_episode=n_episode + 1,
                           num_steps=num_steps_episode)
            num_steps_episode += 1  # Sumamos un paso al episodio

        last_episode = {'episode': environment,
                        'learner': learner}

        if verbose:
            # Imprimimos la información de los episodios
            print('EPISODIO {} - Numero de acciones: {} - Reward: {}'
                  .format(n_episode + 1, num_steps_episode, environment.total_reward))

    print_process_info(last_episode=last_episode)


def print_process_info(last_episode, print_q_table=True, print_best_values_states=True,
                       print_best_actions_states=True, print_steps=True, print_path=True):
    """
    Método que imprime por pantalla los resultados de la ejecución
    """
    if print_q_table:
        print('\nQ_TABLE:')
        last_episode['learner'].print_q_table()

    if print_best_values_states:
        print('\nBEST Q_TABLE VALUES:')
        last_episode['learner'].print_best_values_states()

    if print_best_actions_states:
        print('\nBEST ACTIONS:')
        last_episode['learner'].print_best_actions_states()

    if print_steps:
        print('\nPasos: \n   {}'.format(last_episode['episode'].actions_done))

    if print_path:
        print('\nPATH:')
        last_episode['episode'].print_path_episode()

run_agent(learner=DeepQLearner,
          num_episodes=30,
          discount_factor=0.1,
          ratio_explotacion=0.95,
          max_steps=500,
          verbose=True)