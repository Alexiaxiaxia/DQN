# This code is mainly excerpted from the repo: https://github.com/Kchu/DeepRL_PyTorch.
import os
import pickle
import matplotlib.pyplot as plt


# paths for result log
RESULT_PATH = ['./data/plots/dqn_result_o_Pendulum__4image.pkl','./data/plots/dqn_result_o_Pendulum_1image.pkl',\
               './data/plots/dqn_result_o_Pendulum_DDQN.pkl',
               ]


# model load with check
result = []
for i in range(len(RESULT_PATH)):
    if os.path.isfile(RESULT_PATH[i]):
        pkl_file = open(RESULT_PATH[i],'rb')
        result.append(pickle.load(pkl_file))
        pkl_file.close()
    else:
        print('Can not find:', RESULT_PATH[i])

# plot the figure
print('Load complete!')
print('Plotting the curves!')

plt.plot(range(len(result[0])), result[0], label="Pendulum with 1 image and angle velocity")
plt.plot(range(len(result[1])), result[1], label="Pendulum with 4 image and DDQN")
plt.plot(range(len(result[2])), result[2], label="Pendulum with 1 image and angle velocity")


# plt.plot(range(len(result[1])), result[1], label="IQN")

plt.legend()


plt.xlabel('Iteration times(Thousands)')
plt.ylabel('Score')
plt.tight_layout()
plt.grid()
plt.show()




import os
import pickle
import matplotlib.pyplot as plt

# paths for result log
RESULT_PATH = ['./data/plots/dqn_result_o_Pendulum__4image.pkl', './data/plots/dqn_result_o_Pendulum_1image.pkl',\
               './data/plots/dqn_result_o_Pendulum_DDQN.pkl']

# model load with check
result = []
for i in range(len(RESULT_PATH)):
    if os.path.isfile(RESULT_PATH[i]):
        pkl_file = open(RESULT_PATH[i],'rb')
        result.append(pickle.load(pkl_file))
        pkl_file.close()
    else:
        print('Can not find:', RESULT_PATH[i])

# plot the figure
print('Load complete!')
print('Plotting the curves!')

iterations_per_epoch = 1000  # 假设每个epoch包含1000次迭代

epochs_0 = len(result[0]) // iterations_per_epoch
epochs_1 = len(result[1]) // iterations_per_epoch
epochs_2 = len(result[2]) // iterations_per_epoch

plt.plot(range(epochs_0), [sum(result[0][i:i+iterations_per_epoch])/iterations_per_epoch for i in range(0, len(result[0]), iterations_per_epoch)], label="Pendulum with 1 image and angle velocity")
plt.plot(range(epochs_1), [sum(result[1][i:i+iterations_per_epoch])/iterations_per_epoch for i in range(0, len(result[1]), iterations_per_epoch)], label="Pendulum with 4 image and DDQN")
plt.plot(range(epochs_2), [sum(result[2][i:i+iterations_per_epoch])/iterations_per_epoch for i in range(0, len(result[2]), iterations_per_epoch)], label="Pendulum with 1 image and angle velocity")

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Score')
plt.tight_layout()
plt.grid()
plt.show()