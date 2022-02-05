import test_class
import lcs.optim as opt
import numpy.linalg as la
from casadi import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def print(*args):
    __builtins__.print(*("%.5f" % a if isinstance(a, float) else a
                         for a in args))


# color list
color_list = np.linspace(0, 1, 10)

# ==============================   load the generated LCS system   ==================================
lcs_mats = np.load('random_lcs.npy', allow_pickle=True).item()
n_state = lcs_mats['n_state']
n_lam = lcs_mats['n_lam']
A = lcs_mats['A']
C = lcs_mats['C']
D = lcs_mats['D']
G = lcs_mats['G']
F = lcs_mats['F']
lcp_offset = lcs_mats['lcp_offset']

# ==============================   generate the training data    ========================================
# create the data generator
data_generator = test_class.LCS_learner(n_state, n_lam, A, C, D, G, lcp_offset, stiffness=0)
train_data_size = 1000
train_x_batch = 1 * np.random.uniform(-1, 1, size=(train_data_size, n_state))
train_x_next_batch, train_lam_opt_batch = data_generator.dyn_prediction(train_x_batch, theta_val=[])
mode_percentage, unique_mode_list, mode_frequency_list = test_class.statiModes(train_lam_opt_batch)
print('number of modes:', mode_frequency_list.size)
print('training data mode frequency: ', mode_frequency_list)
# check the mode index
mode_list, train_mode_indices = test_class.plotModes(train_lam_opt_batch)

# =============== plot the training data, each color for each mode  ======================================
# plot dimension index
plot_x_indx = 0
plot_y_indx = 0

plt.figure()
plt.title('True modes marked in (o)')
train_x = train_x_batch[:, plot_x_indx]
train_y = train_x_next_batch[:, plot_y_indx]
plt.scatter(train_x, train_y, c=color_list[train_mode_indices], s=30)

# initialize the plot of the learned learned results
plt.ion()
fig, ax = plt.subplots()
ax.set_title('Learned modes marked in (+)')
train_x = train_x_batch[:, plot_x_indx]
train_y = train_x_next_batch[:, plot_y_indx]
plt.scatter(train_x, train_y, c=color_list[train_mode_indices], s=80, alpha=0.3)
pred_x, pred_y = [], []
sc = ax.scatter(pred_x, pred_y, s=30, marker="+", cmap='paried')
plt.draw()

# ==============================   create the learner object    ========================================
learner = test_class.LCS_learner(n_state, n_lam=n_lam, stiffness=10)
true_theta = vertcat(vec(G), vec(D), lcp_offset, vec(A), vec(C)).full().flatten()

# ====================================================================================================
# doing learning process
curr_theta = 0.1 * np.random.randn(learner.n_theta)
# curr_theta = true_theta + 2 * np.random.randn(learner.n_theta)
mini_batch_size = 100
loss_trace = []
theta_trace = []
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2
for k in range(5000):
    # mini batch dataset
    shuffle_index = np.random.permutation(train_data_size)[0:mini_batch_size]
    x_mini_batch = train_x_batch[shuffle_index]
    x_next_mini_batch = train_x_next_batch[shuffle_index]
    lam_mini_batch = train_lam_opt_batch[shuffle_index]

    # compute the lambda batch
    lam_phi_opt_mini_batch, loss_opt_batch = learner.compute_lambda(x_mini_batch, x_next_mini_batch, curr_theta)

    # compute the gradient
    dtheta, loss, dyn_loss, lcp_loss, dtheta_hessian = \
        learner.gradient_step(x_mini_batch, x_next_mini_batch, curr_theta, lam_phi_opt_mini_batch, second_order=False)

    # store and update
    loss_trace += [loss]
    theta_trace += [curr_theta]
    curr_theta = optimizier.step(curr_theta, dtheta)
    # curr_theta = optimizier.step(curr_theta, dtheta_hessian)

    if k % 100 == 0:
        # on the prediction using the current learned lcs
        pred_x_next_batch, pred_lam_batch = learner.dyn_prediction(train_x_batch, curr_theta)

        # compute the prediction error
        error_x_next_batch = pred_x_next_batch - train_x_next_batch
        relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(train_x_next_batch, axis=1)).mean()

        # compute the predicted mode statistics
        pred_mode_list, pred_mode_indices = test_class.plotModes(pred_lam_batch)

        # plot the learned mode
        pred_x = train_x_batch[:, plot_x_indx]
        pred_y = pred_x_next_batch[:, plot_y_indx]
        sc.set_offsets(np.c_[pred_x, pred_y])
        sc.set_array(color_list[pred_mode_indices])
        fig.canvas.draw_idle()
        plt.pause(0.1)

        print(
            '| iter', k,
            '| loss:', loss,
            '| grad:', norm_2(dtheta),
            '| dyn_loss:', dyn_loss,
            '| lcp_loss:', lcp_loss,
            '| relative_prediction_error:', relative_error,
            '| pred_mode_counts:', len(pred_mode_list),
        )

# save

# on the prediction using the current learned lcs
pred_x_next_batch, pred_lam_batch = learner.dyn_prediction(train_x_batch, curr_theta)
# compute the prediction error
error_x_next_batch = pred_x_next_batch - train_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(train_x_next_batch, axis=1)).mean()
# compute the predicted mode statistics
pred_mode_list, pred_mode_indices = test_class.plotModes(pred_lam_batch)
pred_x = train_x_batch[:, plot_x_indx]
pred_y = pred_x_next_batch[:, plot_y_indx]


np.save('learned', {
    'theta_trace': theta_trace,
    'loss_trace': loss_trace,
    'color_list': color_list,
    'train_x': train_x,
    'train_y': train_y,
    'train_mode_indices': train_mode_indices,
    'pred_y': pred_y,
    'pred_mode_indices': pred_mode_indices,
    'relative_error': relative_error,
    'plot_x_index': plot_x_indx,
    'plot_y_index': plot_x_indx,
})
