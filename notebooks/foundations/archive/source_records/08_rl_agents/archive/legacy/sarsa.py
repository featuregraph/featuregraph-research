import gym
#import agents

env = gym.make('Taxi-v3')
agent = SarsaAgent(states=env.observation_space.n, actions=env.action_space.n)

episodes = 10000
total_steps = 0
total_reward = 0

for episode in range(episodes):
    
    state = env.reset()
    reward = 0
    steps = 0
    done = False
    
    while not done:
        action = agent.act(state)
        if agent.last_action is not None:
            agent.learn(agent.last_state, agent.last_action, agent.last_reward, state, action)
            
        agent.last_action = action
        agent.last_state = state
        
        state, reward, done, info = env.step(action)
        agent.last_reward = reward
        
        steps += 1
        total_reward += reward

    total_steps += steps
    
print("Average timesteps taken: {}".format(total_steps/episodes))
print("Average reward: {}".format(total_reward/episodes))
print("Total reward: {}".format(total_reward))
