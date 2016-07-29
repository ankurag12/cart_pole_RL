import gym
import numpy as np
import time
import queue

env = gym.make('CartPole-v0')
outdir = 'tmp/monitor_logs'
#env.monitor.start(outdir, force=True, seed=0)

#MAX_EPISODES = 500
SOLVING_THRESH_AVG_REWARD = 195     # Over 100 consecutive trials
SOLVING_THRESH_CONSEC_TRIALS = 100
TOLERANCE = 1e-2
GAMMA = 0.995

actions = env.action_space
num_actions = actions.n

states = env.observation_space
states_high = env.observation_space.high
states_low = env.observation_space.low
num_state_variables = states.shape[0]

# The description here https://gym.openai.com/envs/CartPole-v0 says
# episode ends at theta > 15deg, observation_space.high says 24deg,
# and experiments say 12deg

# These values are very specific to the problem and hard-coded
# Need to find out a more generic way to assign these values
state_break_points = np.concatenate(([[2.4, 5.0, 12.0 * np.pi / 180, 500.0 * np.pi / 180]],
                                     [[1.2, 0.5, 6.0 * np.pi / 180, 50.0 * np.pi / 180]],
                                     [[0.6, 0.25, 1.0 * np.pi / 180, 25.0 * np.pi / 180]],
                                     [[-0.6, -0.25, -1.0 * np.pi / 180, -25.0 * np.pi / 180]],
                                     [[-1.2, -0.5, -6.0 * np.pi / 180, -50.0 * np.pi / 180]],
                                     [[-2.4, -5.0, -12.0 * np.pi / 180, -500.0 * np.pi / 180]]), axis=0)

num_states = (state_break_points.shape[0]-1)**state_break_points.shape[1] + 1 # One extra for the "done" state
state_shape = (state_break_points.shape[0]-1) * np.ones(num_state_variables, dtype=int)

# State value function V(s)
v = np.zeros(num_states)

# Initially we have no idea about the state transition probablities Pr(s,a,s')
# Initialize with uniform probability
prob_trans = np.ones((num_states, num_actions, num_states))/num_states

# Reward as a function of state
reward = np.zeros(num_states)
reward_count = np.zeros(num_states)
reward_sum = np.zeros(num_states)

# Policy pi(s)
policy = np.random.randint(num_actions, size=num_states)

count_trans = np.zeros((num_states, num_actions, num_states))
total_count = np.zeros((num_actions, num_states))

num_episodes = 0
reward_history = queue.deque(maxlen=SOLVING_THRESH_CONSEC_TRIALS)
average_reward = 0
reward_curr_ep = 0
problem_solved = False

def sub2ind(array_shape, sub):
    return np.ravel_multi_index(sub, dims=array_shape, order='F')


def obs2state(obs, st_brk_pts, st_shape):
    if obs.shape[0] != st_brk_pts.shape[1]:
        raise ValueError('Number of observations must be equal to number of columns in break points matrix...')
    st = np.zeros(obs.shape[0], dtype=int)
    for i in range(obs.shape[0]):
        st[i] = np.nonzero(np.logical_and((obs[i] < st_brk_pts[0:-1,i]) , (obs[i] >= st_brk_pts[1:,i]) ))[0]
    st_ind = sub2ind(st_shape, st)
    return st_ind


observation = env.reset()
state = obs2state(observation, state_break_points, state_shape)
consecutive_no_learning_trials = 0

while not problem_solved:

    #env.render()

    # To deal with equal values
    # Cannot just select the first index returned by argmax()
    # It has to be randomly selected in case of multiple indices
    temp = np.dot(prob_trans[state, :, :], v)
    action = np.random.choice(np.flatnonzero(temp == temp.max()))

    observation, rew, done, info = env.step(action)
    if not done:
        new_state = obs2state(observation, state_break_points, state_shape)
    else:
        new_state = num_states - 1  # Indexing starts at 0
    #print('observation = ', observation, 'action=', action, 'new_state = ', new_state, 'done=', done)

    count_trans[state, action, new_state] += 1

    reward_count[new_state] += 1
    reward_sum[new_state] += rew
    reward_curr_ep += rew

    if done:
        for i_st in range(num_states):
            for i_act in range(num_actions):
                den = np.sum(count_trans[i_st, i_act, :])
                if den > 0:
                    prob_trans[i_st, i_act, :] = count_trans[i_st, i_act, :] / den

        for i_st in range(num_states):
            if reward_count[i_st] > 0:
                reward[i_st] = reward_sum[i_st] / reward_count[i_st]

        v_new = np.zeros(num_states)
        iter = 0

        while True:

            for i_st in range(num_states):
                v_new[i_st] = np.amax(np.dot(prob_trans[i_st, :, :], v))

            v_new = reward + GAMMA * v_new
            delta = np.linalg.norm(v_new - v, np.inf)
            v[:] = v_new    # Not  v = v_new !!

            if delta < TOLERANCE:
                break

        observation = env.reset()
        state = obs2state(observation, state_break_points, state_shape)

        num_episodes += 1

        reward_history.append(reward_curr_ep)
        #print(reward_curr_ep)
        reward_curr_ep = 0

        # Check if problem solved
        if num_episodes >= SOLVING_THRESH_CONSEC_TRIALS:
            average_reward = sum(reward_history)/SOLVING_THRESH_CONSEC_TRIALS
            print(average_reward)
            if average_reward >= SOLVING_THRESH_AVG_REWARD:
                problem_solved = True

    else:
        state = new_state

print('Optimal policy found in %d episodes' % num_episodes)
#env.monitor.close()

