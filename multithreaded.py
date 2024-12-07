import math
import threading
import queue
from os import write
from time import sleep
import pygame
import matplotlib.pyplot as plt
import numpy as np
import random
from deap import base, creator, tools


WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255)
INSTRUCTION_COUNT = 40
GENERATIONS = 50

check_if_best_car_lock = threading.Lock()
best_fitness = -float('inf')
best_car_steps = []
best_car_counter = 0

class Car:

    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]  # Calculate Center

        self.radars = []
        self.drawing_radars = []

        self.alive = True

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 60
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def eval_car_thread(individual, game_map, screen, clock, result_queue):
    car = Car()
    fitness = 0
    steps = []

    counter = 0
    while car.is_alive() and counter < INSTRUCTION_COUNT * 2:

        choice = individual[counter % INSTRUCTION_COUNT]
        if counter > 0 and counter % INSTRUCTION_COUNT == 0:
            print("overflow of instructions")

        car.speed = 100
        if choice == 0:
            car.angle += 15  # Turn left
        elif choice == 1:
            car.angle -= 15  # Turn right

        if car.is_alive():
            car.update(game_map)
            fitness += car.get_reward()
            steps.append((car.position[:], car.angle))
        counter += 1

    with check_if_best_car_lock:
        global best_fitness, best_car_steps, best_car_counter
        if fitness > best_fitness:
            best_fitness = fitness
            best_car_steps = steps
            best_car_counter = counter

    result_queue.put((fitness, steps))

def threaded_evaluation(individuals, game_map, screen, clock):
    threads = []
    result_queue = queue.Queue()
    for individual in individuals:
        thread = threading.Thread(target=eval_car_thread, args=(individual, game_map, screen, clock, result_queue))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return results

def draw_best_car(screen, game_map, clock):
    car = Car()
    for position, angle in best_car_steps:
        screen.blit(game_map, (0, 0))
        car.position = position
        car.angle = angle
        car.rotated_sprite = car.rotate_center(car.sprite, car.angle)
        car.draw(screen)
        pygame.display.flip()
        clock.tick(300)

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_map = pygame.image.load('map.png').convert()
    clock = pygame.time.Clock()

    # Check if 'FitnessMax' and 'Individual' have already been created
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # 0 - turn left, 1 - turn right - 2 - do nothing
    toolbox.register("attr_int", random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=INSTRUCTION_COUNT)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=6)

    population = toolbox.population(n=250)
    stagnation_count = 0
    best_fitness = None

    # Initialize lists for storing results
    best_fitness_values = [0] * GENERATIONS
    worst_fitness_values = [0] * GENERATIONS
    average_fitness_values = [0] * GENERATIONS
    std_fitness_values = [0] * GENERATIONS

    for gen in range(GENERATIONS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        results = threaded_evaluation(population, game_map, screen, clock)
        for ind, (fitness, _) in zip(population, results):
            ind.fitness.values = (fitness,)

        fitness_values = [ind.fitness.values[0] for ind in population]

        current_best = np.max(fitness_values)
        current_worst = np.min(fitness_values)
        current_average = np.mean(fitness_values)
        current_std = np.std(fitness_values, mean=current_average)

        if best_fitness is None or current_best > best_fitness:
            best_fitness = current_best
            stagnation_count = 0
        else:
            stagnation_count += 1


        mutation_probability = 0.6 if stagnation_count < 8 else 0.8

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring

        draw_best_car(screen, game_map, clock)
        print(f"Generation {gen}, instructions: {best_car_counter}, "
              f"best fitness: {current_best}, worst: {current_worst}, average: {current_average}")
        # Store the fitness values for the current generation
        best_fitness_values[gen] = current_best
        worst_fitness_values[gen] = current_worst
        average_fitness_values[gen] = current_average
        std_fitness_values[gen] = current_std

    pygame.quit()
    return {
        'best_fitness_values': best_fitness_values,
        'worst_fitness_values': worst_fitness_values,
        'average_fitness_values': average_fitness_values,
        'std_fitness_values': std_fitness_values
    }

def plot_results(best_fitness_values, worst_fitness_values, average_fitness_values, std_fitness_values):
    indexes = list(range(1, GENERATIONS + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(indexes, best_fitness_values, color='green', label='Best Fitness')
    plt.plot(indexes, worst_fitness_values, color='red', label='Worst Fitness')
    plt.plot(indexes, average_fitness_values, color='blue', label='Average Fitness')

    # Adding standard deviation
    plt.fill_between(indexes,
                     np.array(average_fitness_values) - np.array(std_fitness_values),
                     np.array(average_fitness_values) + np.array(std_fitness_values),
                     color='blue', alpha=0.2)

    # Adding labels and title
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Values Over Generations')
    plt.legend()

    plt.grid(True)
    # Show plot
    plt.show()


def average_results(results_list):
    avg_best_fitness = np.mean([result['best_fitness_values'] for result in results_list], axis=0)
    avg_worst_fitness = np.mean([result['worst_fitness_values'] for result in results_list], axis=0)
    avg_average_fitness = np.mean([result['average_fitness_values'] for result in results_list], axis=0)
    avg_std_fitness = np.mean([result['std_fitness_values'] for result in results_list], axis=0)

    return {
        'best_fitness_values': avg_best_fitness,
        'worst_fitness_values': avg_worst_fitness,
        'average_fitness_values': avg_average_fitness,
        'std_fitness_values': avg_std_fitness
    }

if __name__ == "__main__":
    results_list = [run_simulation() for _ in range(10)]
    avg_results = average_results(results_list)

    print("Average Best Fitness Values:", avg_results['best_fitness_values'])
    print("Average Worst Fitness Values:", avg_results['worst_fitness_values'])
    print("Average Fitness Values:", avg_results['average_fitness_values'])
    print("Average Std Fitness Values:", avg_results['std_fitness_values'])

    plot_results(**avg_results)

