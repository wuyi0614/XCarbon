# XCarbon - X-Carbon Pricing Decision-making System (X-CarPDS)

The X-Carbon pricing decision-making system aims to help build a carbon pricing system composed by:

- Deep learning price tracker
- Reinforcement learning based carbon pricing decision-making system

## Agent-based modeling (v0.7)
The RL decision-making system has a few components: 

- BaseModule: an underlying agent-based model with economics basis
- RLModule: a reinforcement learning module as optimizer for interaction data from the underlying module
- OptModule: an optimization module with heaps of machine learning tools that learns from the historical outputs
- Scheduler: a scheduler that controls economic underlying models, making outputs, optimization, and retraining
- Interface: a user-friendly interface for simplified interactions between the system and users

## Underlying modules

### Components

#### DONE

- [x] Production
- [x] Energy
- [x] CarbonTrade
- [x] Abatement
- [x] Policy
- [x] Finance

#### TODO

- [ ] Strategy: allow scenario-based settings for the simulation of component/firm/market
- [ ] the expected carbon price should be correlated with the abatement cost, and thus it makes firms randomized in abatement choices
- [ ] the abatement cost should be correlated with firm's carbon intensity, higher the intensity, lower the cost
- [ ] Production: use `carbon intensity` for the initialization of output/allocation in the system
- [ ] Abatement: abatement cost is built in the form of a MAC curve (in `Scheduler.clear_yearly`)

### Firm

#### DONE

- [x] enable different industry specifications including: `power, cement, iron, ...`
- [x] energy module supports **4** types of energy as inputs: `coal, gas, oil, electricity`
- [x] abatement module supports **3** types of technology choices: `efficiency, shift, negative`
- [x] revenue decision of firms and allow firms to quit when they bankrupt
- [x] support fitting with historic data by configuring data in `Scheduler` for `energy, abatement, product`
- [x] build metric-based monitor system to track `output, emission, abatement, allowance allocation`

#### TODO

- [ ] use a monthly adjustable abatement cost as the anchor of carbon price

### Market

#### DONE

- [x] `cap-and-trade` target of the carbon market, default as `45e8 tCO2`
- [x] annual decay rate of the `cap` in the carbon market, default as `0`
- [x] set the entry threshold `68,000 tCO2 (26,000 ton coal)`

#### TODO

- [ ] include `Block Trade` in the market (the threshold is `100,000` tonnes)
- [ ] support real-world market data broadcasting backward into the system

### WebApp

#### DONE

- [x] use streamlit as a web-based interface for XCarbon

#### TODO

- [ ] user management system
- [ ] process parameter submission and simulation with multiprocessing 

## Forthcoming: Sequential modeling
The deep learning price tracker is built with deep learning models and trained with carbon price sequential data. 
It's a typical model that serves to give predictions based on the historical inputs or the other real-time inputs. 

## Contributors:
- Yi Wu, Bartlett School of Sustainable Construction UCL, wymario@163.com (personal)
- Jiaqi Chen, SJTU
