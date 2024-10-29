import random
import time
import math

from matplotlib import pyplot as plt
import numpy as np

# SA parameters
initial_acceptance_probability = 0.15
cooling_rate = 0.99
markov_chain_length = 100

def fitness(solution):
    n = len(solution)
    conflicts = 0

    # Conjuntos para verificar conflitos em diagonais
    diagonal1 = set()  # Diferenca (linha - coluna)
    diagonal2 = set()  # Soma (linha + coluna)

    # Verificar conflitos de linha e diagonais
    for row in range(n):
        col = solution[row]
        
        # Verifica conflito de linha (cada coluna deve ser única)
        if solution.count(col) > 1:
            conflicts += 1
            
        # Verifica conflito de diagonais
        if (row - col) in diagonal1:
            conflicts += 1
        if (row + col) in diagonal2:
            conflicts += 1

        diagonal1.add(row - col)
        diagonal2.add(row + col)

    return conflicts


def initialize_particles(n, num_particles):
    particles = []
    for _ in range(num_particles):
        particle = list(range(n))
        random.shuffle(particle)
        particles.append(particle)
    return particles

def initialize_velocities(n, num_particles):
    velocities = []
    for _ in range(num_particles):
        velocity = [0] * n
        velocities.append(velocity)
    return velocities

def update_velocity_position(particle, velocity, pbest, gbest, W, C1, C2):
    n = len(particle)

    for i in range(n):
        r1 = random.random()
        r2 = random.random()

        # Update the velocity
        velocity[i] = W * velocity[i] + C1 * r1 * (pbest[i] - particle[i]) + C2 * r2 * (gbest[i] - particle[i])

    # Apply sigmoid function to velocities and update positions
    for i in range(n):
        new_pos = (particle[i] + int(velocity[i])) % n 
        particle[i], particle[new_pos] = particle[new_pos], particle[i]

    return particle, velocity

# Select the local global best (nearest neighbor)
def get_local_gbest(particles, fitness_scores, i, neighborhood_size):
    num_particles = len(particles)
    neighbors = []
    
    for j in range(i - neighborhood_size, i + neighborhood_size + 1):
        neighbor_index = j % num_particles
        neighbors.append((particles[neighbor_index], fitness_scores[neighbor_index]))
    
    best_neighbor = min(neighbors, key=lambda x: x[1])
    return best_neighbor[0]

def pso(n, num_particles, max_iterations, neighborhood_size):
    W = 0.5  # Inertia
    C1 = 1.5  # Cognitive coefficient (individual)
    C2 = 2.0  # Social coefficient (global)

    particles = initialize_particles(n, num_particles)
    velocities = initialize_velocities(n, num_particles)

    pbest = [p[:] for p in particles]  # Best individual position
    pbest_fitness = [fitness(p) for p in pbest]
    
    fitness_scores = [fitness(p) for p in particles]

    start_time = time.time()


    for iteration in range(max_iterations):
        for i in range(num_particles):
            
            gbest_local = get_local_gbest(particles, fitness_scores, i, neighborhood_size)

            particles[i], velocities[i] = update_velocity_position(particles[i], velocities[i], pbest[i], gbest_local, W, C1, C2)
            
            current_fitness = fitness(particles[i])
            
            if current_fitness < pbest_fitness[i]:
                pbest[i] = particles[i][:]
                pbest_fitness[i] = current_fitness

            fitness_scores[i] = current_fitness
        
        if min(fitness_scores) == 0:
            break

    # Best global solution found
    best_index = fitness_scores.index(min(fitness_scores))
    return particles[best_index], fitness_scores[best_index], iteration

def sa(initial_state, initial_fitness):
    T = -math.log(initial_acceptance_probability) / (0.001 * initial_fitness)
    current_state = initial_state[:]
    current_fitness = initial_fitness
    best_state = current_state[:]
    best_fitness = current_fitness
    iterations = 0

    while T > 1e-5 and best_fitness > 0:
        for _ in range(markov_chain_length):
            neighbor_state = current_state[:]
            i, j = random.sample(range(len(neighbor_state)), 2)
            neighbor_state[i], neighbor_state[j] = neighbor_state[j], neighbor_state[i]
            neighbor_fitness = fitness(neighbor_state)

            delta = neighbor_fitness - current_fitness

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_state = neighbor_state[:]
                current_fitness = neighbor_fitness

                # Update best found state
                if current_fitness < best_fitness:
                    best_state = current_state[:]
                    best_fitness = current_fitness

                    # Early exit if solution is found
                    if best_fitness == 0:
                        return best_state, best_fitness, iterations
            
            iterations += 1

        # Decrease temperature
        T *= cooling_rate

    return best_state, best_fitness, iterations

def plot_board_with_matplotlib(n, solution):
    board = np.zeros((n, n))
    board[::2, ::2] = 1  # White squares in even positions
    board[1::2, 1::2] = 1  # White squares in odd positions

    # Configure the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(board, cmap='gray', interpolation='nearest')

    for row, col in enumerate(solution):
        x = col
        y = row
        # Draw a red circle representing the queen
        circle = plt.Circle((x, y), 0.4, color='red', fill=True)
        ax.add_patch(circle)

    # Adjust limits and aspect
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert y-axis so position [0,0] is at the bottom-left corner

    plt.show()

def main():
    n = 100
    num_particles = 4
    max_iterations = 1000
    neighborhood_size = 1  
    start_time = time.time()

    # PSO Phase
    best_solution, best_fitness, iterations = pso(n, num_particles, max_iterations, neighborhood_size)
    print("PSO Phase: Best Fitness =", best_fitness)
    print("Best solution from PSO:", best_solution)
    print("Iteration number:", iterations)

    if best_fitness == 0:
        print("Solution Found:")
        print(best_solution)
        plot_board_with_matplotlib(n, best_solution)
        return 
    
    # SA Phase
    best_solution_sa, best_fitness_sa, iterations_sa = sa(best_solution, best_fitness)
    print("SA Phase: Best Fitness =", best_fitness_sa)
    print("Numero de iteraçoes no total:", iterations+iterations_sa)

    if best_fitness_sa == 0:
        print("Solution Found:")
        print(best_solution_sa)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time = {elapsed_time:.2f} seconds")
        
        plot_board_with_matplotlib(n, best_solution_sa)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
