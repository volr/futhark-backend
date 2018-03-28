
# coding: utf-8

# In[2]:


import json
import numpy as np
import subprocess
import sys


# In[55]:


def write_array(file, array):
    def stringify(e):
        if isinstance(e, np.ndarray):
            return "[" + (",".join([stringify(x) for x in e])) + "]"
        else:
            return "{:.1f}".format(e)

    with open(file, 'w') as f:
        f.write(stringify(array))
        f.close()


# In[50]:


# Uses the maze generator at https://github.com/volr/maze
def generate_maze(depth, exit_mask, entry_mask, features):
    command = "maze generate {} [{}] [{}] {}".format(depth, exit_mask, entry_mask, features)
    (maze, _) = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE).communicate()
    return maze


# In[51]:


def maze_to_feature(maze_string):
    maze = json.loads(maze_string)
    left_right = 0 if (maze['left']['node']['tag'] == 'Exit') else 1
    features = np.array(maze['left']['features'], dtype=bool).astype(int)
    return (features, np.array(left_right))


# In[56]:


def generate_data(n, maze_depth, features):
    data = np.array([maze_to_feature(generate_maze(maze_depth, 0, 1, features)) for _ in range(0, n)])
    x = np.array([np.array(x, dtype=int) for x in data[:, 0]])
    y = np.array([np.array(x, dtype=int) for x in data[:, 1]])
    write_array("maze_x_n{}_features{}.txt".format(n, features), x)
    write_array("maze_y_n{}_features{}.txt".format(n, features), y)

generate_data(int(sys.argv[1]), 1, 2)
