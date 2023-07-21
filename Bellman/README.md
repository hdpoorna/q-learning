# Updating a Q-table with the Bellman equation

[Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment was used.

### Requirements
The [following packages](requirements.txt) were used with Python 3.10+.
```
gymnasium[classic-control]
numpy
pandas
matplotlib
```

### Quickstart to Training
_(Assuming the requirements are fulfilled)_
```
python train.py
```
The above commands will do the following.
- Initialize a Q-table from random normal distribution with mean -1.0 and standard deviation 0.5
- Start updating the Q-table with the [default training configurations](helpers/config.py).
- Write the configurations to `results/{QTABLE_ID}/{QTABLE_ID}.txt`
- Save the **rewards** and **exploration rate epsilon** as `npy` files to `results/{QTABLE_ID}/`
- Save the Q-tables as `npy` files to `q_tables/{QTABLE_ID}`

### Quickstart to Testing
_(Assuming the requirements are fulfilled)_
```
python test.py
```
The above commands will load the trained default Q-table and play 1 game.
