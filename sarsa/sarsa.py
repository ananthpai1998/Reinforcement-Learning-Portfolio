import numpy as np

def epsilon_greedy(Q, state, epsilon, nA):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

def run_500_episode(env, policy, discount_factor=0.999, state_discretize=None, max_steps=250):
    total_rewards = []
    total_steps = []
    games_won = 0

    for i in range(500):
        observation, info = env.reset()
        if state_discretize: 
            observation = state_discretize(observation) 
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        while not (terminated or truncated) and step <= max_steps:
            action = int(policy[observation])
            observation, reward, terminated, truncated, info = env.step(action)
            if state_discretize:
                observation = state_discretize(observation)
            total_reward += (discount_factor ** step) * reward
            step += 1

        total_rewards.append(total_reward)
        total_steps.append(step)

        if step > max_steps-2:
            games_won += 1

    avg_reward = sum(total_rewards) / 500
    avg_steps = sum(total_steps) / 500

    return avg_reward, games_won, avg_steps


def run_SARSA(env, total_episodes=15000, max_steps=500, alpha=0.01, gamma=0.999, start_epsilon=1, state_discretize=None):
    # nS, nA = env.observation_space.n, env.action_space.n
    nS, nA = 10000, 2
    Q = np.ones((nS, nA))
    rewards = []
    sub_optimal_policies = {  }

    for episode in range(total_episodes):
        # Linear decay of epsilon
        epsilon = start_epsilon * (total_episodes - episode - 1) / (total_episodes - 1)
        
        state_continuous, info = env.reset()
        state = state_discretize(state_continuous) if state_discretize else state_continuous
        action = epsilon_greedy(Q, state, epsilon, nA)
        
        for step in range(max_steps):
            next_state_continuous, reward, terminated, truncated, info = env.step(action)
            next_state = state_discretize(next_state_continuous) if state_discretize else next_state_continuous
            next_action = epsilon_greedy(Q, next_state, epsilon, nA)

            # SARSA Update, adjust for termination
            if terminated or truncated:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            Q[state, action] += alpha * delta
            
            state, action = next_state, next_action

            if terminated or truncated:
                break        

        # Evaluate policy every 20 episodes
        if (episode + 1) % 20 == 0:
            policy = np.argmax(Q, axis=1)
            avg_reward = run_500_episode(env, policy, gamma, state_discretize=state_discretize, max_steps=max_steps)
            rewards.append(avg_reward)
            print(f"Episode: {episode + 1}, Average Reward: {avg_reward}, Epsilon: {epsilon:.4f}")
            
            if (episode + 1) % 2500 == 0 and episode < 10000 or (episode + 1) % 5000 == 0:
                sub_optimal_policies[f'episode_{episode + 1}'] = policy            

    optimal_policy = np.argmax(Q, axis=1)
    sub_optimal_policies['optimal_policy'] = optimal_policy
    return rewards, sub_optimal_policies
