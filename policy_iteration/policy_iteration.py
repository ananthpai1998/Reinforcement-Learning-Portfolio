import numpy as np


def run_500_episode(env, current_policy, gamma):
    total_rewards = []
    games_won = 0
    steps_per_episode = []
    
    for _ in range(500):
        observation, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0
        
        while not (terminated or truncated):
            action = int(current_policy[observation])
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += (gamma ** step) * reward
            step += 1
            
        if reward == 1:
            games_won += 1
        
        total_rewards.append(total_reward)
        steps_per_episode.append(step)
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    
    return avg_reward, games_won, avg_steps

def bellman_operator(env, state, value_function, policy, gamma):
    total_value = 0
    action = policy[state]
    for outcome in env.P[state][action]:
        transition_probability, next_state, reward, _ = outcome
        outcome_value = transition_probability * (reward + gamma * value_function[next_state])
        total_value += outcome_value
    return total_value

def policy_evaluation(env, policy, value_function, gamma):
    for i in range(500):
        delta = 0
        for state in range(env.observation_space.n):
            v = value_function[state]
            value_function[state] = bellman_operator(env, state, value_function, policy, gamma)
            delta = max(delta, np.abs(v - value_function[state]))
        if delta < 0.001:
            break
    return value_function

def policy_improvement(env, policy, value_function, gamma):
    not_policy_stable = True
    for state in range(env.observation_space.n):
        old_action = policy[state]
        action_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            action_value = 0
            for transition_probability, next_state, reward, _ in env.P[state][action]:
                action_value += transition_probability * (reward + gamma * value_function[next_state])
            action_values[action] = action_value
        best_action = np.argmax(action_values)
        policy[state] = best_action
        if old_action != best_action:
            not_policy_stable = False
        
    return policy, not_policy_stable

def policy_iteration(env, gamma):
    policy = np.zeros(env.observation_space.n , dtype=int)
    value_function = np.zeros(env.observation_space.n)
    is_policy_stable = False
    count = 0
    avg_rewards = []
    games_won_list = []
    avg_steps_list = []

    while not is_policy_stable:
        count += 1
        value_function = policy_evaluation(env, policy, value_function, gamma)
        policy, is_policy_stable  = policy_improvement(env, policy, value_function, gamma)
        max_diff = 0
        for state in range(env.observation_space.n):
            total_value = bellman_operator(env, state, value_function, policy, gamma)
            diff  = np.abs(total_value - value_function[state])
            max_diff = max(max_diff, diff)
        mean_reward, games_won, avg_steps = run_500_episode(env, policy, gamma)
        avg_rewards.append(mean_reward)
        games_won_list.append(games_won)
        avg_steps_list.append(avg_steps)
        print(f'Iteration {count}, Mean Reward: {mean_reward}, Games Won: {games_won}, Avg Steps: {avg_steps}')
        if max_diff < 0.001:
            break
    return  value_function, avg_rewards, games_won_list, avg_steps_list, policy