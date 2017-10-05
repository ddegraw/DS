import kagglegym

env = kagglegym.make()
observation = env.reset()

print(len(observation.target))
print(len(observation.train))


train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)


ids = train["id"].unique()
means = {}

for i in ids:
    ys = train[train["id"] == i].y
    means[i] = ys.mean()
    

rewards = []
i = 0
while True:
    n = 0
    randy =[]
    sym = list(observation.target.id)
    target = observation.target
    
    #test_x = np.array(observation.features[col].values).reshape(-1,1)
    
    nosyms =[item for item in sym if item not in means.keys()]
    for i in nosyms:
        means[i] = 0.002
    
    #sym = list(observation.target.id)
    #observation.target.y = [means[d] for d in sym]
    #randy = random.sample(list(means.values()),sym)
    randy = [means[d] for d in sym]

        
            
    target.loc[:,'y'] = randy
    #observation.target.fillna(0, inplace=True)

    observation, reward, done, info = env.step(target)  
    if done:
        break
    rewards.append(reward)
    i = i + 1
    #print "Iteration #", i
    #print "#Rewards", len(rewards)    

print(info)
print(n)
print(len(rewards))