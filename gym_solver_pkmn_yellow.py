import environment_pyboy_neat_pkmn_yellow as emt
import argparse
import numpy as np
from neat import nn, population, statistics, parallel
import neat
import pickle
import gc


### User Params ###

# The name of the game to solve
game_name = 'GbaGame'
### End User Params ###


parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=1000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--render', action='store_true')
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--num-cores', dest="numCores", type=int, default=4,
                    help="The number cores on your computer for parallel execution")
args = parser.parse_args()


def preprocess_observation(obs):
    # Assuming obs is a NumPy array of shape (height, width, num_frames)
    # Flatten the observation to a 1D array
    flattened_obs = obs.flatten()

    # Optionally, normalize the pixel values from 0-255 to 0-1
    normalized_obs = flattened_obs / 255.0

    return normalized_obs


def simulate_species(net, env, episodes=1, steps=5000, render=False):
    fitnesses = []
    for _ in range(episodes):
        obs, _ = env.reset()
        cum_reward = 0.0
        for _ in range(steps):
            # Preprocess the observation
            processed_obs = preprocess_observation(obs)

            # Activate the network with the processed observation
            outputs = net.activate(processed_obs)

            # Decide on an action based on the network output
            action = np.argmax(outputs)
            obs, reward, done, trunc, _ = env.step(action)
            if render:
                env.render()
            if done:
                break
            cum_reward += reward

        fitnesses.append(cum_reward)

    fitness_avg = np.mean(fitnesses)
    fitness_max = np.max(fitnesses)
    gc.collect()

    # print("Species avg fitness: %s" % str(fitness_avg))
    # print("Species max fitness: %s" % str(fitness_max))
    return fitness_max


def worker_evaluate_genome(g, config):
    my_env = emt.GbaGame()
    net = nn.feed_forward.FeedForwardNetwork.create(g, config)
    return simulate_species(net, my_env, args.episodes, args.max_steps, render=args.render)


def train_network(env):
    def evaluate_genome(g, config):  # Modified to include config
        net = nn.feed_forward.FeedForwardNetwork.create(g, config)
        return simulate_species(net, env, args.episodes, args.max_steps, render=args.render)

    def eval_fitness(genomes, config):  # Modified to include config
        for genome_id, g in genomes:  # Adjusted to unpack genome_id and genome
            fitness = evaluate_genome(g, config)  # Pass config to evaluate_genome
            g.fitness = fitness
            print(genome_id, g.fitness)

    # Simulation setup remains the same...
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, filename='config-feedforward_pkmn_yellow')
    pop = population.Population(cfg)
    # Load checkpoint, parallel execution setup, and so on remain unchanged...
    # Add a stdout reporter to show progress in the terminal and a statistics reporter for detailed statistics
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    checkpointer = neat.Checkpointer(generation_interval=10, filename_prefix='neat-checkpoint-')
    pop.add_reporter(checkpointer)
    # Important change: pass 'config' to 'eval_fitness' when running simulation
    if args.render:
        pop.run(eval_fitness, args.generations)
    else:
        pe = parallel.ParallelEvaluator(args.numCores, worker_evaluate_genome)
        pop.run(pe.evaluate, args.generations)

        # Optionally, you can manually save a checkpoint after training is done
        checkpointer.save_checkpoint(cfg, pop, pop.species, pop.generation)

    # Show output of the most fit genome against training data.
    winner = pop.best_genome

    # Save best network
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    winner_net = nn.feed_forward.FeedForwardNetwork.create(winner, cfg)
    for i in range(20):
        simulate_species(winner_net, env, 1, args.max_steps, render=True)


if __name__ == '__main__':
    train_network(emt.GbaGame())
