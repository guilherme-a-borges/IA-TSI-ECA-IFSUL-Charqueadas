import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ============================================================
# 1. Universes
# ============================================================

x_servico = np.arange(0, 10.1, 0.1)
x_comida = np.arange(0, 10.1, 0.1)
x_gorjeta = np.arange(0, 25.1, 0.1)

# ============================================================
# 2. Membership functions
# ============================================================

servico_ruim = fuzz.trapmf(x_servico, [0, 0, 2, 5])
servico_aceitavel = fuzz.trimf(x_servico, [2, 5, 8])
servico_excelente = fuzz.trapmf(x_servico, [5, 8, 10, 10])

comida_ruim = fuzz.trapmf(x_comida, [0, 0, 2, 5])
comida_aceitavel = fuzz.trimf(x_comida, [2, 5, 8])
comida_saborosa = fuzz.trapmf(x_comida, [5, 8, 10, 10])

gorjeta_baixa = fuzz.trapmf(x_gorjeta, [0, 0, 4, 10])
gorjeta_media = fuzz.trapmf(x_gorjeta, [6, 10, 14, 18])
gorjeta_alta = fuzz.trapmf(x_gorjeta, [14, 20, 25, 25])

methods = ['centroid', 'mom', 'som', 'lom']
method_names = {
    'centroid': 'Centroid',
    'mom': 'MOM',
    'som': 'SOM',
    'lom': 'LOM'
}

colors = {
    'centroid': 'red',
    'mom': 'blue',
    'som': 'green',
    'lom': 'purple'
}

# ============================================================
# 3. Fuzzy inference function
# ============================================================

def fuzzy_inference(servico_value, comida_value):
    # Fuzzification
    s_ruim = fuzz.interp_membership(x_servico, servico_ruim, servico_value)
    s_aceitavel = fuzz.interp_membership(x_servico, servico_aceitavel, servico_value)
    s_excelente = fuzz.interp_membership(x_servico, servico_excelente, servico_value)

    c_ruim = fuzz.interp_membership(x_comida, comida_ruim, comida_value)
    c_aceitavel = fuzz.interp_membership(x_comida, comida_aceitavel, comida_value)
    c_saborosa = fuzz.interp_membership(x_comida, comida_saborosa, comida_value)

    # Rules
    rule_baixa = max(s_ruim, c_ruim)
    rule_media = max(s_aceitavel, c_aceitavel)
    rule_alta = max(s_excelente, c_saborosa)

    # Implication
    out_baixa = np.fmin(rule_baixa, gorjeta_baixa)
    out_media = np.fmin(rule_media, gorjeta_media)
    out_alta = np.fmin(rule_alta, gorjeta_alta)

    # Aggregation
    aggregated = np.fmax(out_baixa, np.fmax(out_media, out_alta))

    # Defuzzification
    results = {}
    for method in methods:
        results[method] = fuzz.defuzz(x_gorjeta, aggregated, method)

    activations = {
        'Low tip': rule_baixa,
        'Medium tip': rule_media,
        'High tip': rule_alta
    }

    return aggregated, results, activations


# ============================================================
# 4. Initial values
# ============================================================

initial_servico = 7
initial_comida = 7
selected_method = 'centroid'

aggregated, results, activations = fuzzy_inference(initial_servico, initial_comida)

# ============================================================
# 5. Figure layout
# ============================================================

fig, axs = plt.subplots(2, 2, figsize=(13, 8))
plt.subplots_adjust(left=0.10, bottom=0.25, right=0.85)

ax_servico = axs[0, 0]
ax_comida = axs[0, 1]
ax_gorjeta = axs[1, 0]
ax_bar = axs[1, 1]

# ============================================================
# 6. Static plots
# ============================================================

def draw_input_plot(ax, x, mf1, mf2, mf3, labels, value, title):
    ax.clear()
    ax.plot(x, mf1, label=labels[0])
    ax.plot(x, mf2, label=labels[1])
    ax.plot(x, mf3, label=labels[2])
    ax.axvline(value, color='black', linestyle='--', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Input value')
    ax.set_ylabel('Membership degree')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def update_plot():
    global aggregated, results, activations

    servico_value = slider_servico.val
    comida_value = slider_comida.val
    method = radio.value_selected.lower()

    aggregated, results, activations = fuzzy_inference(servico_value, comida_value)

    # Service plot
    draw_input_plot(
        ax_servico,
        x_servico,
        servico_ruim,
        servico_aceitavel,
        servico_excelente,
        ['Poor', 'Acceptable', 'Excellent'],
        servico_value,
        'Service'
    )

    # Food plot
    draw_input_plot(
        ax_comida,
        x_comida,
        comida_ruim,
        comida_aceitavel,
        comida_saborosa,
        ['Poor', 'Acceptable', 'Tasty'],
        comida_value,
        'Food'
    )

    # Output plot
    ax_gorjeta.clear()
    ax_gorjeta.plot(x_gorjeta, gorjeta_baixa, '--', label='Low tip', alpha=0.5)
    ax_gorjeta.plot(x_gorjeta, gorjeta_media, '--', label='Medium tip', alpha=0.5)
    ax_gorjeta.plot(x_gorjeta, gorjeta_alta, '--', label='High tip', alpha=0.5)

    ax_gorjeta.fill_between(
        x_gorjeta,
        np.zeros_like(x_gorjeta),
        aggregated,
        color='orange',
        alpha=0.45,
        label='Aggregated output'
    )

    for m in methods:
        linewidth = 3 if m == method else 1.5
        alpha = 1.0 if m == method else 0.45

        ax_gorjeta.axvline(
            results[m],
            color=colors[m],
            linestyle='--',
            linewidth=linewidth,
            alpha=alpha,
            label=f'{method_names[m]} = {results[m]:.2f}%'
        )

    ax_gorjeta.set_title(f'Defuzzification result: {method_names[method]} = {results[method]:.2f}%')
    ax_gorjeta.set_xlabel('Tip (%)')
    ax_gorjeta.set_ylabel('Membership degree')
    ax_gorjeta.set_ylim(-0.05, 1.05)
    ax_gorjeta.legend(loc='upper right', fontsize=8)
    ax_gorjeta.grid(True, alpha=0.3)

    # Bar plot comparing methods
    ax_bar.clear()
    values = [results[m] for m in methods]
    bar_colors = [colors[m] if m == method else 'lightgray' for m in methods]

    ax_bar.bar([method_names[m] for m in methods], values, color=bar_colors)
    ax_bar.set_title('Comparison between methods')
    ax_bar.set_ylabel('Tip (%)')
    ax_bar.set_ylim(0, 25)
    ax_bar.grid(True, axis='y', alpha=0.3)

    for i, value in enumerate(values):
        ax_bar.text(i, value + 0.5, f'{value:.2f}', ha='center')

    # Rule activation text
    fig.suptitle(
        f'Service = {servico_value:.1f} | Food = {comida_value:.1f} | '
        f'Rule activations: Low={activations["Low tip"]:.2f}, '
        f'Medium={activations["Medium tip"]:.2f}, '
        f'High={activations["High tip"]:.2f}',
        fontsize=12
    )

    fig.canvas.draw_idle()


# ============================================================
# 7. Interactive controls
# ============================================================

ax_slider_servico = plt.axes([0.15, 0.13, 0.55, 0.03])
slider_servico = Slider(
    ax=ax_slider_servico,
    label='Service',
    valmin=0,
    valmax=10,
    valinit=initial_servico,
    valstep=0.1
)

ax_slider_comida = plt.axes([0.15, 0.08, 0.55, 0.03])
slider_comida = Slider(
    ax=ax_slider_comida,
    label='Food',
    valmin=0,
    valmax=10,
    valinit=initial_comida,
    valstep=0.1
)

ax_radio = plt.axes([0.88, 0.35, 0.10, 0.25])
radio = RadioButtons(
    ax_radio,
    ['centroid', 'mom', 'som', 'lom'],
    active=0
)

slider_servico.on_changed(lambda val: update_plot())
slider_comida.on_changed(lambda val: update_plot())
radio.on_clicked(lambda label: update_plot())

# First draw
update_plot()

plt.show()