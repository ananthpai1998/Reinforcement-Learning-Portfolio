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

def modified_policy_iteration(env, gamma=0.999, max_iterations=500, m=3):
    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.random.choice(nA, nS)
    V = np.zeros(nS)
    avg_rewards = []
    games_won_list = []
    avg_steps_list = []

    for i in range(max_iterations):
        for _ in range(m):
            prev_V = np.copy(V)
            for s in range(nS):
                a = policy[s]
                expected_value = 0
                for probability, next_state, reward, _ in env.P[s][a]:
                    expected_value += probability * (reward + gamma * prev_V[next_state])
                    V[s] = expected_value
            if np.max(np.abs(prev_V - V)) <= 0.001:
                break
        policy_stable = True
        for s in range(nS):
            old_action = policy[s]
            action_values = np.zeros(nA)
            for a in range(nA):
                for p, s_, r, _ in env.P[s][a]:
                    action_values[a] += p * (r + gamma * V[s_])
            new_action = np.argmax(action_values)
            policy[s] = new_action
            if old_action != new_action:
                policy_stable = False
        
        if i % 20 == 0:
            print(i)
            mean_reward, games_won, avg_steps = run_500_episode(env, policy, gamma)
            avg_rewards.append(mean_reward)
            games_won_list.append(games_won)
            avg_steps_list.append(avg_steps)
            print(f'Iteration {i}, Mean Reward: {mean_reward}, Games Won: {games_won}, Avg Steps: {avg_steps}')
        
        # if policy_stable:
        #     break
    return V, avg_rewards, games_won_list, avg_steps_list, policy



