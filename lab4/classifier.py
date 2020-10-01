from tqdm import tqdm
import numpy as np
import scipy.special as scipy
from tqdm import tqdm


LEARNING_RATE = 0.1
N_EPOCHS = 5
RANDOM_SEED = 42


class NeuralNetwork:
    """
    Нейронная сеть прямого распространения.

    Parameters
    ----------
    input_nodes : int
        Количество входных нейронов сети.
    hidden_nodes : tuple
        Описание скрытых слоёв сети в виде кортежа.
        Например, (100, 100) означает 2 скрытых слоя по 100 нейронов в каждом.
    out_nodes : int
        Количество выходных нейронов сети.
    lr : int
        Коэффицент обучения. Рекомендация: 0 < lr < 1.
    """
    def __init__(self,
        input_nodes: int,
        hidden_nodes: tuple,
        out_nodes: int,
        lr=LEARNING_RATE,
    ):
        self.input_nodes = input_nodes
        self.hidden_layers_num = len(hidden_nodes)
        self.hidden_nodes = hidden_nodes
        self.out_nodes = out_nodes
        self.lr = lr

        # 3-х мерная матрица весов всех связей
        self.weights = []
        # 3-х мерная матрица значений выхода всех узлов
        self.outs  = []
        # 3-х мерная матрица ошибок для каждого узла
        self.errors = []

        self.__weights_init()
    
    def __weights_init(self):
        """
        Начальная инициализация всех весов
        """
        # Для связей между входными нейронами и первым слоем скрытых нейронов
        link = (self.hidden_nodes[0], self.input_nodes)
        self.weights.append(
            np.random.normal(0.0, self.hidden_nodes[0] ** (-0.5), link)
        )
        # Для дальнейших связей между слоями скрытых нейронах
        bufL = 0
        layer = 0
        for layer in range(self.hidden_layers_num - 1):
            bufL += 1
            link = ( self.hidden_nodes[layer + 1], self.hidden_nodes[layer] )
            self.weights.append(
                np.random.normal(0.0, self.hidden_nodes[layer + 1] ** (-0.5), link)
            )
        # для конечной связи между выходными нейронами и последним слоем скрытых
        link = (self.out_nodes, self.hidden_nodes[bufL]) \
            if layer == 0 else (self.out_nodes, self.hidden_nodes[layer + 1])
        self.weights.append(
            np.random.normal(0.0, self.out_nodes ** (-0.5), link)
        )

    def error(self):
        """
        Метод для определения среднеквадратичной ошибки.
        Чем меньше этот показатель, тем мы ближе к эталонной точности.
        """
        return sum(er*er for er in self.errors[0])

    def fit(self,
        X,
        Y,
    ):
        """
        Метод для обучения нейронной сети с учителем

        Parameters
        ----------
        x : array-like
            Входные данные в виде массива, размерностью (1, input_nodes)
        y : array-like
            Метка для выходных данных х, размерностью (1, out_nodes)
        """
        self.errors.clear()
        
        # Пробуем предсказать
        test_out = self.predict(X)

        # Преобразование данных к нужному виду
        X = np.array(X, ndmin=2).T
        Y = np.array(Y, ndmin=2).T

        # Вычисление взвешенных ошибок на каждом узле сети, 
        # начиная с сравнения выхода сети с Y.
        self.errors.append(
            np.asarray(Y - test_out)
        )
        # Обратное распространение ошибки по всем остальным слоям сети
        for i in range(self.hidden_layers_num):
            self.errors.append( 
                np.dot(
                    self.weights[self.hidden_layers_num - i].T,
                    self.errors[i],
                )
        )
        
        # Коррекция весов сети, на основе найденной ошибки
        # методом градиентного спуска. Нужно искать минимум функции, 
        # в нашем случае, это будет минимум ошибки на кажом узле.
        for i in range(self.hidden_layers_num + 1):
            self.weights[self.hidden_layers_num - i] += \
                self.lr * np.dot(
                    self.errors[i] * self.outs[self.hidden_layers_num - i] * (1 - self.outs[self.hidden_layers_num - i]),
                    np.transpose(self.outs[self.hidden_layers_num - i - 1]) \
                        if self.hidden_layers_num - i - 1 >= 0 else np.transpose(X)
                )

    def predict(self, X):
        """
        Предсказание сети
        
        Parameters
        ----------
        x : array-like
            Входные данные для предсказания, размерностью (1, input_nodes)
        """
        X = np.array(X, ndmin=2).T
        self.outs.clear()

        # Получение значения выхода на первом слое скрытых нейронов,
        # который непосредственно связан со входными значениями.
        out = np.dot(self.weights[0], X)
        self.outs.append(
            np.asarray( scipy.expit(out) )
        )

        # Последующиее вычисления значений на нейронах по слоям
        for i in range(len(self.hidden_nodes)):
            out = np.dot(self.weights[i + 1], self.outs[i])
            self.outs.append(
                np.asarray( scipy.expit(out) )
            )

        # Последний слой -- результирующий
        return self.outs[-1]
