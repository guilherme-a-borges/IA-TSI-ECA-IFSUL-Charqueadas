import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Genetic Algorithm for PID Tuning
# Context: IoT-based smart temperature control system
# ==========================================================

# -----------------------------
# Process model parameters
# -----------------------------
AMBIENT_TEMP = 25.0       # °C
INITIAL_TEMP = 25.0       # °C
SETPOINT = 60.0           # desired temperature °C

TIME_STEPS = 250
DT = 1.0                  # seconds

THERMAL_GAIN = 0.8        # how much the heater affects temperature
THERMAL_INERTIA = 40.0    # higher value = slower system

MAX_HEATER_POWER = 100.0  # actuator limit in %


# -----------------------------
# GA parameters
# -----------------------------
POPULATION_SIZE = 30
GENERATIONS = 50
MUTATION_RATE = 0.20
ELITE_SIZE = 2

# Bounds for PID parameters
KP_RANGE = (0.0, 20.0)
KI_RANGE = (0.0, 2.0)
KD_RANGE = (0.0, 10.0)


# ==========================================================
# PID-controlled system simulation
# ==========================================================
def simulate_pid(kp, ki, kd):
    temperature = INITIAL_TEMP
    integral = 0.0
    previous_error = SETPOINT - temperature

    temperatures = []
    heater_powers = []
    errors = []

    for t in range(TIME_STEPS):
        error = SETPOINT - temperature
        integral += error * DT
        derivative = (error - previous_error) / DT

        # PID control law
        heater_power = kp * error + ki * integral + kd * derivative

        # Actuator saturation: heater can only work from 0% to 100%
        heater_power = max(0.0, min(MAX_HEATER_POWER, heater_power))

        # Simple thermal system model
        # Temperature tends to ambient, but heater increases it
        temperature_change = (
            -(temperature - AMBIENT_TEMP) + THERMAL_GAIN * heater_power
        ) / THERMAL_INERTIA

        temperature += temperature_change * DT

        temperatures.append(temperature)
        heater_powers.append(heater_power)
        errors.append(error)

        previous_error = error

    return np.array(temperatures), np.array(heater_powers), np.array(errors)


# ==========================================================
# Fitness function
# ==========================================================
def fitness(individual):
    kp, ki, kd = individual

    temperatures, heater_powers, errors = simulate_pid(kp, ki, kd)

    # Integral Absolute Error: total accumulated error
    iae = np.sum(np.abs(errors))

    # Overshoot: how much the temperature exceeds the setpoint
    overshoot = max(0.0, np.max(temperatures) - SETPOINT)

    # Control effort: penalizes excessive actuator usage
    effort = np.sum(heater_powers) / TIME_STEPS

    # Final error: penalizes not reaching the target
    final_error = abs(SETPOINT - temperatures[-1])

    # Cost function: lower is better
    cost = iae + 80 * overshoot + 20 * final_error + 0.5 * effort

    # Fitness: higher is better
    return 1 / (1 + cost)


# ==========================================================
# Genetic Algorithm functions
# ==========================================================
def create_individual():
    kp = random.uniform(*KP_RANGE)
    ki = random.uniform(*KI_RANGE)
    kd = random.uniform(*KD_RANGE)
    return [kp, ki, kd]


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


def tournament_selection(population, tournament_size=3):
    candidates = random.sample(population, tournament_size)
    return max(candidates, key=fitness)


def crossover(parent1, parent2):
    # Arithmetic crossover
    alpha = random.random()

    child1 = [
        alpha * parent1[i] + (1 - alpha) * parent2[i]
        for i in range(3)
    ]

    child2 = [
        alpha * parent2[i] + (1 - alpha) * parent1[i]
        for i in range(3)
    ]

    return child1, child2


def mutate(individual):
    kp, ki, kd = individual

    if random.random() < MUTATION_RATE:
        kp += random.uniform(-2.0, 2.0)

    if random.random() < MUTATION_RATE:
        ki += random.uniform(-0.2, 0.2)

    if random.random() < MUTATION_RATE:
        kd += random.uniform(-1.0, 1.0)

    # Keep values inside allowed limits
    kp = max(KP_RANGE[0], min(KP_RANGE[1], kp))
    ki = max(KI_RANGE[0], min(KI_RANGE[1], ki))
    kd = max(KD_RANGE[0], min(KD_RANGE[1], kd))

    return [kp, ki, kd]


# ==========================================================
# Main GA loop
# ==========================================================
def genetic_algorithm():
    population = create_population()

    best_history = []

    for generation in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)

        best = population[0]
        best_fit = fitness(best)
        best_history.append(best_fit)

        print(
            f"Generation {generation:02d} | "
            f"Best fitness = {best_fit:.6f} | "
            f"Kp = {best[0]:.3f}, Ki = {best[1]:.3f}, Kd = {best[2]:.3f}"
        )

        new_population = []

        # Elitism: preserve the best individuals
        new_population.extend(population[:ELITE_SIZE])

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    best = max(population, key=fitness)

    return best, best_history


# ==========================================================
# Run experiment
# ==========================================================
best_pid, best_history = genetic_algorithm()

kp, ki, kd = best_pid

print("\nBest PID parameters found:")
print(f"Kp = {kp:.4f}")
print(f"Ki = {ki:.4f}")
print(f"Kd = {kd:.4f}")

temperatures, heater_powers, errors = simulate_pid(kp, ki, kd)

# ==========================================================
# Plot results
# ==========================================================
time = np.arange(TIME_STEPS) * DT

plt.figure(figsize=(10, 5))
plt.plot(time, temperatures, label="Temperature")
plt.axhline(SETPOINT, linestyle="--", label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Control using GA-Optimized PID")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, heater_powers, label="Heater Power")
plt.xlabel("Time (s)")
plt.ylabel("Heater Power (%)")
plt.title("Actuator Signal")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(best_history)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Genetic Algorithm Evolution")
plt.grid(True)
plt.show()