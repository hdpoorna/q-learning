# Grid World

A [custom environment](gridWorld.py) was made.
- Agent (A): Blue
- Goal (G): Green
- Wall (W): Red (optional)

### Environment

| ![](results/gw-conv-through-2023-07-18-17-32-23/2023-07-21-12-29-52.gif) | ![](results/gridWorld.png) |
|:------------------------------------------------------------------------:|:--------------------------:|


Look into the [Graphs](#graphs) section for more info.

### Requirements
To install,
```
pip install -r requirements.txt
```
The [following packages](requirements.txt) were used with Python 3.10+. Feel free to experiment with different versions.
```
opencv-python==4.8.0.74
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
tensorflow==2.13.0
tqdm==4.65.0
```

### Quickstart to Training
_(Assuming the requirements are fulfilled)_
```
python train.py
```
The above commands will do the following.
- Initialize 2 identical [neural networks](helpers/dqn_helper.py?plain=1#L38) (policy, target) with 3 conv layers.
- Start training policy net and updating target net with the [default training configurations](helpers/config.py).
- Write the model summary and the configurations to `results/{MODEL_ID}/{MODEL_ID}.txt`
- Save the **rewards**, **exploration rate (epsilon)** and **losses** as `npy` files to `results/{MODEL_ID}/`
- Save the policy model as `tf savedModel` to `saved_models/{MODEL_ID}/`

### Quickstart to Testing
_(Assuming the requirements are fulfilled)_
```
python test.py
```
The above commands will do the following.
- Load the trained default saved model.
- Play 1 game and visualize.
- Save the video as a `gif` to `results/{QTABLE_ID}/`

### Graphs

|                Stat                |                               Plot                                |
|:----------------------------------:|:-----------------------------------------------------------------:|
| episode reward <br/>(moving stats) |   ![](results/gw-conv-through-2023-07-18-17-32-23/rewards.svg)    |
|          exploration rate          | ![](results/gw-conv-through-2023-07-18-17-32-23/explorations.svg) |
|     loss <br/>(episode stats)      |     ![](results/gw-conv-through-2023-07-18-17-32-23/loss.svg)     |

It seems the model can be further trained. However, [Google Colab](https://colab.research.google.com) compute time limits had to be considered.

### Notes
- [gridWorld.py](gridWorld.py) can be used to initially explore the environment.
- [Deque](helpers/dqn_helper.py?plain=1#L20) (Double Ended Queue) and [Named-tuple](helpers/dqn_helper.py?plain=1#L15) data structures were used to record the transitions into a buffer.
- [plot_graphs.py](helpers/plot_graphs.py) can be used to plot graphs from the `npy` files, and save them as `svg`.
- You can print the environment by [printing the GridWorld instance](gridWorld.py?plain=1#L271-L274).
    ```
    |_|_|_|_|_|_|_|_|
    |_|_|_|_|_|_|_|_|
    |_|_|_|_|_|_|_|_|
    |_|A|_|_|_|_|_|_|
    |_|_|_|_|_|_|_|_|
    |_|_|_|_|_|_|_|_|
    |_|_|_|_|_|_|_|G|
    |_|_|_|_|_|_|W|_|
    ```

### Play
You can play by using the [play](gridWorld.py?plain=1#L194) method in `GridWorld` class.
- [Initialize](gridWorld.py?plain=1#L271-L279) the environment.
- Set the `MODE` to `PLAY`, if you're using `gridWorld.py` file itself as the entry point.
- Call `play` method.
- key: action map (clockwise)
  - w: up
  - d: right
  - s: down
  - a: left
  - esc: quit
