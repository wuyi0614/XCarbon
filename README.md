# XCarbon - X-Carbon Pricing Decision-making System (X-CarPDS)

The X-Carbon pricing decision-making system aims to help build a carbon pricing system composed by:

- Deep learning price tracker
- Reinforcement learning based carbon pricing decision-making system

## Sequential modeling
The deep learning price tracker is built with deep learning models and trained with carbon price sequential data. 
It's a typical model that serves to give predictions based on the historical inputs or the other real-time inputs. 


## Agent-based modeling
The RL decision-making system has a few components: 

- BaseModule: an underlying agent-based model with economics basis
- RLModule: a reinforcement learning module as optimizer for interaction data from the underlying module
- OptModule: an optimization module with heaps of machine learning tools that learns from the historical outputs
- Scheduler: a scheduler that controls economic underlying models, making outputs, optimization, and retraining
- Interface: a user-friendly interface for simplified interactions between the system and users

## TODOs
The project gradually develops the following features to make the system extensive and flexible for future development.

### Underlying modules

#### Firm level

- specify firms from different industries including `power, cement, iron, ...`
- energy module including various types of `energy input`
- abatement technology including various technologies that could affect `emission factors`
- abatement technology doesn't reduce emission directly but it reduces intermediate use of energy
- revenue decision of firms and allow firms to quit when they bankrupt

#### Market level

- `cap-and-trade` target of the carbon market
- annual decay rate of the `cap` in the carbon market
- set the entry threshold `26,000 tCO2`

### Web UI modules

- explore streamlit as a web-based interface for XCarbon

## Contributors:
- Yi Wu, Bartlett School of Sustainable Construction UCL, wymario@163.com (personal)
