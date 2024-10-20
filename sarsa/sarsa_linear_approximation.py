import numpy as np
from tqdm import tqdm

def q_value(weights, state, action, phi):
    return np.dot(phi(state), weights[action])

def epsilon_greedy_action(state, epsilon, actions, weights, phi):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    q_values = [q_value(weights, state, action, phi) for action in actions]
    return actions[np.argmax(q_values)]

def run_500_episode(env, weights, max_steps, discount_factor, phi, actions):
    total_rewards = []
    total_steps = []
    games_won = 0

    for i in range(500):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0
        while not (terminated or truncated) and step <= max_steps:
            action = actions[np.argmax([q_value(weights, state, action, phi) for action in actions])]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += (discount_factor ** step) * reward
            step += 1
        total_rewards.append(total_reward)
        total_steps.append(step)

        # if step > max_steps-2:
        #     games_won += 1

        if reward > 0:
            games_won += 1
        
    return sum(total_rewards)/500, games_won, sum(total_steps)/500


def sarsa_linear_approximation(env, nA, dim, episodes, alpha, gamma, start_epsilon, max_steps_per_episode, phi, mean, std_dev):
    num_actions = nA
    feature_vector_length = dim
    weights = np.random.normal(mean, std_dev, size=(num_actions, feature_vector_length))
    actions = range(num_actions)
    sub_optimal_weights = {  }

    for episode in tqdm(range(episodes)):
        epsilon = start_epsilon * (episodes - episode - 1) / (episodes - 1)
        
        state, info = env.reset()
        action = epsilon_greedy_action(state, epsilon, actions, weights, phi)

        for _ in range(max_steps_per_episode):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = epsilon_greedy_action(next_state, epsilon, actions, weights, phi)

            if terminated:
                weights[action] += alpha * (reward - q_value(weights, state, action, phi)) * phi(state)

            else:
                td_target = reward + gamma * q_value(weights, next_state, next_action, phi)
                weights[action] += alpha * (td_target - q_value(weights, state, action, phi)) * phi(state)
            
            state, action = next_state, next_action

            if terminated or truncated:
                break

        if episode % (episodes/20) == 0 or episode==49999 or episode==399999:
            # policy = [np.argmax([q_value(weights, state, action, phi) for action in actions]) for state in range(feature_vector_length)]
            mean_reward, games_won, avg_steps = run_500_episode(env, weights, max_steps=max_steps_per_episode, discount_factor=gamma, phi=phi, actions=actions)
            sub_optimal_weights[f"Episode_{episode}"] = weights
            print(f"Episode: {episode}, mean_reward: {mean_reward}, games_won: {games_won}, avg_steps: {avg_steps} , epsilon: {epsilon}")
    sub_optimal_weights["Optimal Policy"] = weights
    return mean_reward, sub_optimal_weights