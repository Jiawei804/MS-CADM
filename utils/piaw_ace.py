import numpy as np
from scipy.stats import norm, t
import pickle
import matplotlib.pyplot as plt


def calculate_ace(true_values, predictions, standard_errors, df=None):
    """
    Calculate the Average Coverage Error (ACE) at different confidence levels.

    Parameters:
    - true_values: Array of true values (shape [n]).
    - predictions: Array of predicted values (shape [n]).
    - standard_errors: Array of standard errors (shape [n]).
    - df: Degrees of freedom (if using t-distribution). If None, the normal distribution is used.

    Returns:
    - ace_results: Dictionary where keys are confidence levels and values are the corresponding ACE.
    """
    # Define the range of confidence levels
    confidence_levels = np.arange(0.1, 1.0, 0.1)
    ace_results = {}

    for cl in confidence_levels:
        # Calculate the significance level
        alpha = 1 - cl
        # Calculate the critical value
        if df is not None:
            # Use t-distribution
            critical_value = t.ppf(1 - alpha / 2, df)
        else:
            # Use normal distribution
            critical_value = norm.ppf(1 - alpha / 2)

        # Calculate the prediction interval
        lower_bound = predictions - critical_value * standard_errors
        upper_bound = predictions + critical_value * standard_errors

        # Calculate the coverage proportion
        coverage = np.mean((true_values >= lower_bound) & (true_values <= upper_bound))

        # Calculate ACE
        ace = coverage - cl
        ace_results[f"{int(cl * 100)}%"] = ace

    return ace_results


def calculate_piaw(predictions, standard_errors, df=None):
    """
    Calculate the Prediction Interval Average Width (PIAW) at different confidence levels.

    Parameters:
    - predictions: Array of predicted values (shape [n]).
    - standard_errors: Array of standard errors (shape [n]).
    - df: Degrees of freedom (if using t-distribution). If None, the normal distribution is used.

    Returns:
    - piaw_results: Dictionary where keys are confidence levels and values are the corresponding PIAW.
    """
    # Define the range of confidence levels
    confidence_levels = np.arange(0.1, 1.0, 0.1)
    piaw_results = {}

    for cl in confidence_levels:
        # Calculate the significance level
        alpha = 1 - cl
        # Calculate the critical value
        if df is not None:
            # Use t-distribution
            critical_value = t.ppf(1 - alpha / 2, df)
        else:
            # Use normal distribution
            critical_value = norm.ppf(1 - alpha / 2)

        # Calculate the prediction interval width
        lower_bound = predictions - critical_value * standard_errors
        upper_bound = predictions + critical_value * standard_errors
        interval_widths = upper_bound - lower_bound

        # Calculate the average width
        piaw = np.mean(interval_widths)
        piaw_results[f"{int(cl * 100)}%"] = piaw

    return piaw_results


if __name__ == "__main__":
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


    models = ['NF', 'GAN', 'VAE', 'CDiT']
    pi_aw_data = dict()
    ace_data = dict()
    for model, pred in zip(models, pred_list):
        pi_aw_data[model] = calculate_piaw(pred, 0.1 * np.std(pred)).values()
        ace_data[model] = calculate_ace(true_values, pred, 0.1 * np.std(pred)).values()

    confidence = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
    markers = ['o', 's', 'x', 'd']
    colors = ['#1f77b4', '#c991f7', '#2ca02c', '#ff7f0e']

    # PIAW
    for i, model in enumerate(models):
        ax1.plot(confidence, pi_aw_data[model], label=f'{model}', marker=markers[i], color=colors[i], markersize=5,
                 linestyle='-', linewidth=1)
    ax1.set_title('(a) PIAW Results', fontsize=10)
    ax1.set_xlabel('Confidence Level', fontsize=10)
    ax1.set_ylabel('PIAW', fontsize=10)
    ax1.set_xticklabels(confidence, fontsize=8)
    ax1.set_yticklabels(np.round(np.linspace(0, 0.5, 6), 2), fontsize=8)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 0.5)
    ax1.legend()

    # ACE
    for i, model in enumerate(models):
        ax2.plot(confidence, ace_data[model], label=f'{model}', marker=markers[i], color=colors[i], markersize=5,
                 linestyle='-', linewidth=1)
    ax2.set_title('(b) ACE Results', fontsize=10)
    ax2.set_xlabel('Confidence Level', fontsize=10)
    ax2.set_ylabel('ACE', fontsize=10)
    ax2.set_xticklabels(confidence, fontsize=8)
    ax2.set_yticklabels(np.round(np.arange(-0.2, 0.01, 0.05), 2), fontsize=8)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(-0.2, 0.01)
    ax2.set_yticks(np.arange(-0.2, 0.01, 0.05))
    ax2.legend()

    plt.tight_layout()
    plt.savefig('PIAW_ACE.png')
    plt.show()