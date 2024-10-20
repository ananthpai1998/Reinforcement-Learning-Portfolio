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

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=500):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V = np.zeros(num_states)
    policy = np.zeros([num_states, num_actions])
    # iteration_count = 0
    
    avg_rewards = []
    games_won_list = []
    avg_steps_list = []

    # while True:
    for iteration_count in range(max_iterations):
        delta = 0
        for s in range(num_states):
            v = V[s]
            action_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
            V[s] = max(action_values)
            delta = max(delta, np.abs(v - V[s]))

        # iteration_count += 1

        if iteration_count % 10 == 0:
            policy = extract_policy(env, V, gamma)
            avg_reward, games_won, avg_steps = run_500_episode(env, np.argmax(policy, axis=1), gamma)
            avg_rewards.append(avg_reward)
            games_won_list.append(games_won)
            avg_steps_list.append(avg_steps)
            print(f"Iteration {iteration_count}: Avg Reward: {avg_reward:.4f}, Games Won: {games_won}, Avg Steps: {avg_steps:.2f}")

        # if delta < theta:
        #     break

    return V, avg_rewards, games_won_list, avg_steps_list, policy

def extract_policy(env, V, gamma=0.99):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.zeros([num_states, num_actions])

    for s in range(num_states):
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    return policy
