import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


with open('scenarios/scenarios_wind_UMNN_M_1_0_100_TEST.pickle', 'rb') as file:
    pred_nf = pickle.load(file)

with open('scenarios/scenarios_wind_GAN_wasserstein_1_0_100_TEST.pickle', 'rb') as file:
    pred_gan = pickle.load(file)

with open('scenarios/scenarios_wind_VAElinear_1_0_100_TEST.pickle', 'rb') as file:
    pred_vae = pickle.load(file)

pred_cdit = np.load('scenarios/pred_wind.npy').transpose(2, 0, 1).reshape(100, -1).transpose(1, 0)

true_values = np.load('scenarios/true_wind.npy').flatten()


print(pred_nf.shape, pred_gan.shape, pred_vae.shape, pred_cdit.shape, true_values.shape)
pred_list = [pred_nf.copy(), pred_gan.copy(), pred_vae.copy(), pred_cdit.copy()]

np.random.seed(42)
idx = np.random.randint(0, 12000 - -24, 1)[0]
idx = 0
pred_nf, pred_gan, pred_vae, pred_cdit, true_values = pred_nf[idx:idx + 72], pred_gan[idx:idx + 24], pred_vae[idx:idx + 24], pred_cdit[idx:idx + 24], true_values[idx:idx + 24]
# pred_nf, pred_gan, pred_vae = pred_nf.transpose(1, 0), pred_gan.transpose(1, 0), pred_vae.transpose(1, 0)

print(pred_nf.shape, pred_gan.shape, pred_vae.shape, pred_cdit.shape, true_values.shape)


N_q= 99
q_set = [i / (N_q + 1) for i in range(1, N_q + 1)]
# Quantiles are generated into an array of shape (n_day*24, N_q)
q_nf = np.quantile(pred_nf, q=q_set, axis=1).transpose()
q_gan = np.quantile(pred_gan, q=q_set, axis=1).transpose()
q_vae = np.quantile(pred_vae, q=q_set, axis=1).transpose()
q_cdit = np.quantile(pred_cdit, q=q_set, axis=1).transpose()
print(q_nf.shape, q_gan.shape, q_vae.shape, q_cdit.shape)
q_test = [q_nf, q_gan, q_vae, q_cdit]

pred_list = [pred_nf, pred_gan, pred_vae, pred_cdit]
f, axes = plt.subplots(2, 2, figsize=(12, 7))
for i, ax in enumerate(axes.flat):

    for j in range(pred_list[i].shape[1]):
        ax.plot(pred_list[i][:, j], linewidth=0.5, color='gray', alpha=0.6)
    ax.plot(true_values, linewidth=1.5, color='red')
    ax.plot(np.mean(pred_list[i], axis=1), linewidth=1.5, color='orange')
    ax.plot(q_test[i][:, 9], color='b', linewidth=1.5)
    ax.plot(q_test[i][:, 89], color='k', linewidth=1.5)

    ax.set_title(f"({chr(97+i)}) {['NF', 'GAN', 'VAE', 'CDiT'][i]}", fontsize=13)
    ax.set_xlabel('time/h', fontsize=12)
    ax.set_ylabel("Power/p.u.", fontsize=12)
    # ax.legend(prop={'size': 12}, loc='upper right')

    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1)
    # ax.legend()

handles = [
    plt.Line2D([0], [0], color='red', lw=1.5, label='Measured Wind Power'),
    plt.Line2D([0], [0], color='orange', lw=1.5, label='Mean Forecast'),
    plt.Line2D([0], [0], color='b', lw=1.5, label='10 %'),
    plt.Line2D([0], [0], color='k', lw=1.5, label='90 %'),
    plt.Line2D([0], [0], color='gray', lw=1.5, label='Wind Power Scenarios'),
]
f.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False, prop={'size': 12}, columnspacing=10,)

plt.tight_layout()
plt.savefig('scenarios.pdf')
plt.show()