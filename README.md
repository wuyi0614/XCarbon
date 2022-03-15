# XCarbon - X-Carbon Decision Support System (X-CDSS)

## Introduction

The X-Carbon decision support system aims to build a carbon market decision support system for carbon price prediction, which is composed by:

- Deep learning price tracker
- Agent-based carbon market modelling
- Reinforcement learning supported decision-making system
- Version = 0.8

## Agent-based modeling
The Decision Support System is built on an agent-based model with: 

- Components: the elements for firms and the market, including Production, Energy, Abatement, CarbonTrade, Policy, Finance
- Firm: the customized firm types (e.g. Regulated Firm) for the market
- Market: the customized market object for the carbon market, including an OrderBook object
- Scheduler: a scheduler that controls economic underlying models, making outputs, optimization, and retraining
- WebApp: a user-friendly interface for necessary interactions between the system and users (supported by streamlit)
- RLModule (forthcoming): a reinforcement learning module as optimizer for interaction data from the underlying module

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
- [ ] Abatement: the abatement cost should be correlated with firm's carbon intensity, higher the intensity, lower the cost
- [x] Production: use `carbon intensity` for the initialization of output/allocation in the system
- [ ] Abatement: abatement cost is built in the form of a MAC curve (in `Scheduler.clear_yearly`)
- [ ] CarbonTrade: the expected carbon price should be correlated with the abatement cost, and thus it makes the firms randomized in abatement choices
- [ ] CarbonTrade: obtain expected carbon price by optimizing firm's abatement cost and supply/demand equilibrium in the market

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
