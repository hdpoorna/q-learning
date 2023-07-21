# Updating a Q-table with the Bellman equation

[Gymnasium Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) environment was used.

### Results

|                             default                              |                              greedy                              |
|:----------------------------------------------------------------:|:----------------------------------------------------------------:|
| ![](results/default-2023-07-13-08-51-31/2023-07-21-08-18-03.gif) | ![](results/exploit-2023-07-12-05-19-03/2023-07-21-08-19-21.gif) |

### Requirements
The [following packages](requirements.txt) were used with Python 3.10+.
```
gymnasium[classic-control]
opencv-python
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
The above commands will do the following.
- Load the trained default Q-table.
- Play 1 game and visualize.
- Save the video as a `gif` to `results/{QTABLE_ID}`

### Notes
- [test_env.py](helpers/test_env.py) can be used to initially explore the environment.
- [play_mt_car.py](helpers/play_mt_car.py) can be used to control the car with the following keys and play.
  - a: left
  - s: do nothing
  - d: right
- [plot_graphs.py](helpers/plot_graphs.py) can be used to plot graphs from the saved `rewards.npy` and `explorations.npy` files.
