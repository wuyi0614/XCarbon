# XCarbon - X-Carbon Pricing Decision-making System (X-CarPDS)

The X-Carbon pricing decision-making system aims to help build a carbon pricing system composed by:

- Deep learning price tracker
- Reinforcement learning based carbon pricing decision-making system

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

##### DONE

- [x] enable different industry specifications including: `power, cement, iron, ...`
- [x] energy module supports **4** types of energy as inputs: `coal, gas, oil, electricity`
- [x] abatement module supports **3** types of technology choices: `efficiency, shift, negative`
- [x] abatement cost is built in the form of a MAC curve (in `Scheduler.update_yearly_price`)
- [x] revenue decision of firms and allow firms to quit when they bankrupt

##### TODO

- [ ] use a monthly adjustable abatement cost as the anchor of carbon price
- [ ] support fitting with historic data by configuring data in `Scheduler` for `energy, abatement, product`
- [ ] build metric-based monitor system to track `output, emission, abatement, allowance allocation`

#### Market level

##### DONE

- [x] `cap-and-trade` target of the carbon market
- [x] annual decay rate of the `cap` in the carbon market
- [x] set the entry threshold `26,000 tCO2`

##### TODO

- [ ] include `Block Trade` in the market (the threshold is `100,000` tonnes)
- [ ] support real-world market data broadcasting backward into the system


### Web UI modules

##### DONE

- [x] use streamlit as a web-based interface for XCarbon

##### TODO

- [ ] user management system
- [ ] process parameter submission and simulation with multiprocessing 

## TODO: Sequential modeling
The deep learning price tracker is built with deep learning models and trained with carbon price sequential data. 
It's a typical model that serves to give predictions based on the historical inputs or the other real-time inputs. 

## Contributors:
- Yi Wu, Bartlett School of Sustainable Construction UCL, wymario@163.com (personal)
- Jiaqi Chen, SJTU
