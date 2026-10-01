import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Variables
servico = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'servico')
comida = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'comida')
gorjeta = ctrl.Consequent(np.arange(0, 25.1, 0.1), 'gorjeta')

# 2. Input membership functions
servico['ruim'] = fuzz.trapmf(servico.universe, [0, 0, 2, 5])
servico['aceitavel'] = fuzz.trimf(servico.universe, [2, 5, 8])
servico['excelente'] = fuzz.trapmf(servico.universe, [5, 8, 10, 10])

comida['ruim'] = fuzz.trapmf(comida.universe, [0, 0, 2, 5])
comida['aceitavel'] = fuzz.trimf(comida.universe, [2, 5, 8])
comida['saborosa'] = fuzz.trapmf(comida.universe, [5, 8, 10, 10])

# 3. Output membership functions
# The trapezoids create flat maximum regions, making SOM, MOM and LOM differ.
gorjeta['baixa'] = fuzz.trapmf(gorjeta.universe, [0, 0, 4, 10])
gorjeta['media'] = fuzz.trapmf(gorjeta.universe, [6, 10, 14, 18])
gorjeta['alta'] = fuzz.trapmf(gorjeta.universe, [14, 20, 25, 25])

# 4. Rules
regra1 = ctrl.Rule(servico['ruim'] | comida['ruim'], gorjeta['baixa'])
regra2 = ctrl.Rule(servico['aceitavel'] | comida['aceitavel'], gorjeta['media'])
regra3 = ctrl.Rule(servico['excelente'] | comida['saborosa'], gorjeta['alta'])

sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])

# 5. Function to compute with a selected defuzzification method
def calcular_gorjeta(valor_servico, valor_comida, metodo):
    gorjeta.defuzzify_method = metodo

    # Important: create a fresh simulation to avoid cached results
    simulador = ctrl.ControlSystemSimulation(sistema_controle, cache=False)

    simulador.input['servico'] = valor_servico
    simulador.input['comida'] = valor_comida
    simulador.compute()

    return simulador.output['gorjeta'], simulador


# 6. Comparison
metodos = ['centroid', 'mom', 'som', 'lom']
nomes = ['Centroid', 'Mean of Maximum - MOM', 'Smallest of Maximum - SOM', 'Largest of Maximum - LOM']

entradas_teste = [
    (3, 7),
    (5, 5),
    (7, 7),
    (8, 6),
    (9, 4)
]

for s, c in entradas_teste:
    print(f"\nInput: Service = {s}, Food = {c}")
    print("-" * 55)

    for metodo, nome in zip(metodos, nomes):
        valor, _ = calcular_gorjeta(s, c, metodo)
        print(f"{nome:28}: {valor:.2f}%")


# 7. Plot one didactic example
servico_exemplo = 7
comida_exemplo = 7

resultados = {}

for metodo in metodos:
    valor, simulador_plot = calcular_gorjeta(servico_exemplo, comida_exemplo, metodo)
    resultados[metodo] = valor

# Plot aggregated output using one simulation
gorjeta.view(sim=simulador_plot)

# Add vertical lines for each defuzzification result
cores = {
    'centroid': 'red',
    'mom': 'blue',
    'som': 'green',
    'lom': 'purple'
}

for metodo, valor in resultados.items():
    plt.axvline(
        valor,
        color=cores[metodo],
        linestyle='--',
        linewidth=2,
        label=f'{metodo}: {valor:.2f}%'
    )

plt.title(f'Defuzzification comparison: Service={servico_exemplo}, Food={comida_exemplo}')
plt.legend()
plt.show()