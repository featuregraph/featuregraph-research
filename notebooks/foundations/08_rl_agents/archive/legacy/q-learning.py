import gym
#import agents

env = gym.make('Taxi-v3')
agent = QLearningAgent(states=env.observation_space.n, actions=env.action_space.n)

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
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, _)
        state = next_state
        
        steps += 1
        total_reward += reward

    total_steps += steps
    
print("Average timesteps taken: {}".format(total_steps/episodes))
print("Average reward: {}".format(total_reward/episodes))
print("Total reward: {}".format(total_reward))
