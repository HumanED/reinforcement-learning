import numpy as np
import matplotlib.pyplot as plt
thumb_high = np.array([0.960, 1.222, 0.209, 0.436, 1.571])
thumb_low = np.array([-0.960, 0, -0.209, -0.436, 0])
thumb_discrete_actions = np.array([1,3,8,10,0])
thumb_bin_size = (thumb_high - thumb_low) / 11
thumb_cont_actions = thumb_low + (thumb_bin_size / 2) + (thumb_bin_size * thumb_discrete_actions)
index = 4
plt.hist(np.arange(thumb_low[index],thumb_high[index],0.001),11,edgecolor='k')
plt.scatter(x=thumb_cont_actions[index],y=50,color='red')
plt.show()