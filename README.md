# XCarbon - X-Carbon Pricing Decision-making System (X-CarPDS)

The X-Carbon pricing decision-making system aims to help build a carbon pricing system composed by:

- Deep learning price tracker
- Reinforcement learning based carbon pricing decision-making system

## Price Tracker
The deep learning price tracker is built with deep learning models and trained with carbon price sequential data. 
It's a typical model that serves to give predictions based on the historical inputs or the other real-time inputs. 

## RL decision-making system
The RL decision-making system has a few components: 

- BaseModule: an underlying agent-based model with economics basis
- RLModule: a reinforcement learning module as optimizer for interaction data from the underlying module
- OptModule: an optimization module with heaps of machine learning tools that learns from the historical outputs
- Scheduler: a scheduler that controls economic underlying models, making outputs, optimization, and retraining
- Interface: a user-friendly interface for simplified interactions between the system and users

## Contributors:
- Yi Wu, Bartlett School of Sustainable Construction UCL, wymario@163.com (personal)
