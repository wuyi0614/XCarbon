# Instructions about how to play with X-Carbon core modules

## Conceptual designs

### Component activation

#### 1. How to initiate the instances?

**Check codes in the `component.py` at the bottom**

- read configuration `config` file from `config/<industry>-firm-<timestamp>.json`

```python
config = read_config('config/power-firm-20220206.json')
```

- initiate `Clock` by specifying a start date.

```python
clock = Clock('2021-07-16')
```

- initiate `Energy` first with `config['Energy']` configurations.

```python
energy = Energy(random=False, **config['Energy'])
```

- initiate `CarbonTrade` with `Clock, Energy, Factors, config['CarbonTrade']`.

```python
carbon = CarbonTrade(clock=clock, Energy=energy.snapshot(), uid='buyer',
                    Factors=config['Production']['Factors'], **config['CarbonTrade'])

```

#### 2. How does each instance update after initiation (yearly)?

- pass `Finance.Profit` down to the next-year production plan

