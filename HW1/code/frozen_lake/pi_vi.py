#! python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium

import lake_info



def value_func_to_policy(env, gamma, value_func):
    '''
    Outputs a policy given a value function.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute the policy for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.

    Returns
    -------
    np.ndarray
        An array of integers. Each integer is the optimal action to take in
        that state according to the environment dynamics and the given value
        function.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    # BEGIN STUDENT SOLUTION
    for s in range(len(value_func)):
        action_rewards = {}
        for a in range(env.action_space.n):
            r_array = env.unwrapped.P[s][a]
            action_rewards[a] = sum([p * (value_func[s_]+r) for p, s_, r, _ in r_array])
        policy[s] = max(action_rewards, key=action_rewards.get)
    # END STUDENT SOLUTION
    return(policy)



def evaluate_policy_sync(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    while delta > tol:
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            a = policy[s]
            r_array = env.unwrapped.P[s][a]
            value_func[s] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            delta = max(delta, abs(v - value_func[s]))
        i += 1
    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    while (delta > tol and i < max_iters):
        new_vf = np.zeros_like(value_func)
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            a = policy[s]
            r_array = env.P[s][a]
            new_vf[s] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            delta = max(delta, abs(v - new_vf[s]))
        i += 1
        value_func = new_vf
    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a policy. Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    while (delta > tol and i < max_iters):
        new_vf = np.zeros_like(value_func)
        delta = 0
        for s in np.random.permutation(env.observation_space.n):
            v = value_func[s]
            a = policy[s]
            r_array = env.P[s][a]
            new_vf[s] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            delta = max(delta, abs(v - new_vf[s]))
        i += 1
        value_func = new_vf
    # END STUDENT SOLUTION
    return(value_func, i)



def improve_policy(env, gamma, value_func, policy):
    '''
    Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.
    policy: np.ndarray
        The policy to improve, maps states to actions.

    Returns
    -------
    (np.ndarray, bool)
        Returns the new policy and whether the policy changed.
    '''
    policy_changed = False
    # BEGIN STUDENT SOLUTION
    for s in range(len(value_func)):
        old_action = policy[s]
        action_rewards = {}
        for a in range(env.action_space.n):
            r_array = env.unwrapped.P[s][a]
            action_rewards[a] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
        policy[s] = max(action_rewards, key=action_rewards.get)
        if policy[s] != old_action:
            policy_changed = True
    # END STUDENT SOLUTION
    return(policy, policy_changed)



def policy_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    changed = True
    while changed:
        value_func, iterations = evaluate_policy_sync(env, value_func, gamma, policy)
        pe_steps += iterations
        policy, changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    changed = True
    while (changed):
        value_func, iters = evaluate_policy_async_ordered(env, value_func, gamma, policy)
        policy, changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        pe_steps += iters
        if(pi_steps >= max_iters):
            break
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)


def policy_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    changed = True
    while (changed):
        value_func, iters = evaluate_policy_async_randperm(env, value_func, gamma, policy)
        policy, changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        pe_steps += iters
        if(pi_steps >= max_iters):
            break
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def value_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    i = 0
    delta = 1
    while i < max_iters and delta > tol:
        delta = 0
        for s in range(env.observation_space.n):
            if value_func[s] > 0:
                continue
            v = value_func[s]
            action_rewards = []
            for a in range(env.action_space.n):
                r_array = env.unwrapped.P[s][a]
                action_reward = 0
                for p, s_, r, done in r_array:
                    if done:
                        if r > 0:
                            value_func[s_] = r
                        action_reward = max(action_reward, r)  # Max reward if terminal state
                    else:
                        action_reward = max(action_reward, p * (r + gamma * value_func[s_]))
                action_rewards.append(action_reward)
            max_action_reward = max(action_rewards)
            value_func[s] = max_action_reward
            delta = max(delta, abs(v - value_func[s]))
        i += 1
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    while (delta >= tol and i < max_iters):
        delta = 0
        for s in range(env.observation_space.n):
            v = value_func[s]
            action_space = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                r_array = env.P[s][a]
                action_space[a] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            value_func[s] = max(action_space)
            delta = max(delta, abs(v - value_func[s]))
        i += 1
        value_func = value_func
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    while (delta >= tol and i < max_iters):
        delta = 0
        for s in np.random.permutation(env.observation_space.n):
            v = value_func[s]
            action_space = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                r_array = env.P[s][a]
                action_space[a] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            value_func[s] = max(action_space)
            delta = max(delta, abs(v - value_func[s]))
        i += 1
        value_func = value_func
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_custom(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    #create state ordering using manhattan distance to goal
    state_order = np.zeros(env.observation_space.n, dtype='int')
    #get goal state
    goal = np.unravel_index(env.unwrapped.desc.ravel().argmax(), env.unwrapped.desc.shape)
    for s in range(env.observation_space.n):
        state = np.unravel_index(s, env.unwrapped.desc.shape)
        state_order[s] = abs(goal[0] - state[0]) + abs(goal[1] - state[1])
    state_order = np.argsort(state_order)
    print(state_order)
    # reverse state order
    state_order = state_order[::-1]

    while (delta >= tol and i < max_iters):
        delta = 0
        for s in state_order:
            v = value_func[s]
            action_space = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                r_array = env.P[s][a]
                action_space[a] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            value_func[s] = max(action_space)
            delta = max(delta, abs(v - value_func[s]))
        i += 1
        value_func = value_func
    # END STUDENT SOLUTION
    return(value_func, i)

    """value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    delta = 1
    i = 0
    state_count = env.observation_space.n - 1
    while (delta >= tol and i < max_iters):
        delta = 0
        new_vf = np.zeros_like(value_func)
        for s in range(state_count, -1, -1):
            v = value_func[s]
            action_space = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                r_array = env.P[s][a]
                action_space[a] = sum([p * (r + gamma * value_func[s_]) for p, s_, r, _ in r_array])
            new_vf[s] = max(action_space)
            delta = max(delta, abs(v - new_vf[s]))
    # END STUDENT SOLUTION
    return(value_func, i)"""



# Here we provide some helper functions for your convinience.

def display_policy_letters(env, policy):
    '''
    Displays a policy as an array of letters.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    policy: np.ndarray
        The policy to display, maps states to actions.
    '''
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_info.actions_to_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.unwrapped.nrow, env.unwrapped.ncol)

    for row in range(env.unwrapped.nrow):
        print(''.join(policy_letters[row, :]))



def value_func_heatmap(env, value_func):
    '''
    Visualize a policy as a heatmap.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    value_func: np.ndarray
        The current value function estimate.
    '''
    fig, ax = plt.subplots(figsize=(7,6))

    # Reshape value_func to match the environment dimensions
    heatmap_data = np.reshape(value_func, [env.unwrapped.nrow, env.unwrapped.ncol])

    # Create a heatmap using Matplotlib
    cax = ax.matshow(heatmap_data, cmap='GnBu_r')

    # Set ticks and labels
    ax.set_yticks(np.arange(0, env.unwrapped.nrow))
    ax.set_xticks(np.arange(0, env.unwrapped.ncol))
    ax.set_yticklabels(np.arange(1, env.unwrapped.nrow + 1)[::-1])
    ax.set_xticklabels(np.arange(1, env.unwrapped.ncol + 1))

    # Display the colorbar
    cbar = plt.colorbar(cax)

    plt.show()

def run(agent, env, max_steps):
    observation, info = env.reset()
    episode_observations, episode_actions, episode_rewards = [], [], []

    for _ in range(max_steps):
        action = agent[observation]

        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_observations.append(observation)
        episode_actions.append(action)
        episode_rewards.append(reward)

        observation = next_observation

        if terminated:
            break

    return(episode_observations, episode_actions, episode_rewards)

if __name__ == '__main__':
    np.random.seed(10003)
    maps = lake_info.maps
    gamma = 0.9
    for map_name, map in maps.items():
        env = gymnasium.make('FrozenLake-v1', desc=map, map_name=map_name, is_slippery=False, render_mode='human')
        # BEGIN STUDENT SOLUTION
        # do policy iteration sync
        #p, v, pi, pe = policy_iteration_async_ordered(env, gamma)
        #print(pi, pe)
        v, i = value_iteration_async_custom(env, gamma)
        p = value_func_to_policy(env, gamma, v)
        #run(p, env, 20)
        print(i)
        #policy = np.zeros(env.observation_space.n, dtype='int')
        # display policy
        #print(display_policy_letters(env, p))
        #value_func_heatmap(env, v)
        #print(p)
        # END STUDENT SOLUTION
