class Neuron():
    def __init__(
        self, 
        n_input=2,
        func='limit',
        threshold=1,
    ):
        self.weights = [1 for _ in range(n_input)]
        self.func = func
        self.threshold = threshold
    
    def set_weights(self, weights):
        self.weights = weights

    def calc_func(self, s):
        out = 0
        if self.func == 'limit':
            out = 1 if s >= self.threshold else 0
        elif self.func == 'sigmoid':
            out = 1 if s >= self.threshold else 0
        return out

    def predict(self, inputs):
        s = 0
        formula = ''
        result = ''
        print(f'In: {inputs}')
        for n, i, w in zip(range(len(inputs)), inputs, self.weights):
            s += i*w
            formula += f'i_{n}*w_{n} + '
            result += f'{i}*{w} + '
        pred = self.calc_func(s)
        print(f's = {formula[:-3]} = {result[:-2]} = {s}, F(s) = {pred}')
        return pred

if __name__=='__main__':
    neuron = Neuron()
    neuron.set_weights([1, 1])
    values = [[0, 0], [0, 1], [1, 0], [1, 1]]

    print('(1) Logical AND:')
    neuron.threshold = 1
    for value in values:
        neuron.predict(value)

    print('(2) Logical OR:')
    neuron.threshold = 2
    for value in values:
        neuron.predict(value)
    
    print('(3) Logical NAND:')
    neuron.set_weights([-1, -1])
    neuron.threshold = -1
    for value in values:
        neuron.predict(value)