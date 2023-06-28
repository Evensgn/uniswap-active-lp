import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_x_tau_for_diff_risk_a(dir_path, hi_lo=None):
    results = np.load(os.path.join(dir_path, 'results_merge.npy'), allow_pickle=True).item()
    risk_averse_a_list = results['risk_averse_a_list']
    tau_list = results['tau_list'][:-1]
    static_tau = results['tau_list'][-1]
    fix_tau = 20
    results = results['results']

    if not os.path.exists(os.path.join(dir_path, 'plots')):
        os.mkdir(os.path.join(dir_path, 'plots'))
    dir_path = os.path.join(dir_path, 'plots')

    overlapping_alpha = 0.9

    nn_best_tau_expected_wealth_list = []
    nn_best_tau_std_wealth_list = []

    vector_best_tau_expected_wealth_list = []
    vector_best_tau_std_wealth_list = []

    static_expected_wealth_list = []
    static_std_wealth_list = []

    for risk_averse_a in risk_averse_a_list:
        mean_rebalance_list = []
        utility_nn_list = []
        utility_vector_list = []
        utility_uniform_dollar_list = []
        utility_uniform_liq_list = []

        expected_wealth_nn_list = []
        expected_wealth_vector_list = []
        expected_wealth_uniform_dollar_list = []
        expected_wealth_uniform_liq_list = []

        std_wealth_nn_list = []
        std_wealth_vector_list = []
        std_wealth_uniform_dollar_list = []
        std_wealth_uniform_liq_list = []

        for tau in tau_list:
            job_key = 'job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, tau)

            eval_results = results[job_key]['eval_results']
            mean_rebalance_list.append(eval_results['mean_rebalance_count'])

            utility_nn_list.append(eval_results['expected_utility_nn'])
            utility_vector_list.append(eval_results['expected_utility_vector'])
            utility_uniform_dollar_list.append(eval_results['expected_utility_uniform_dollar'])
            utility_uniform_liq_list.append(eval_results['expected_utility_uniform_liq'])

            expected_wealth_nn_list.append(np.mean(eval_results['wealth_nn_list']))
            expected_wealth_vector_list.append(np.mean(eval_results['wealth_vector_list']))
            expected_wealth_uniform_dollar_list.append(np.mean(eval_results['wealth_uniform_dollar_list']))
            expected_wealth_uniform_liq_list.append(np.mean(eval_results['wealth_uniform_liq_list']))

            std_wealth_nn_list.append(np.std(eval_results['wealth_nn_list']))
            std_wealth_vector_list.append(np.std(eval_results['wealth_vector_list']))
            std_wealth_uniform_dollar_list.append(np.std(eval_results['wealth_uniform_dollar_list']))
            std_wealth_uniform_liq_list.append(np.std(eval_results['wealth_uniform_liq_list']))

            nn_portions_mean = eval_results['nn_portions_mean']
            solved_single_vector = results[job_key]['solved_single_vector']
            print('job_key={}'.format(job_key))
            print('nn_portions_mean={}'.format(nn_portions_mean))
            print('solved_single_vector={}'.format(solved_single_vector))

        job_key_static = 'job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, static_tau)
        utility_static = results[job_key_static]['eval_results']['expected_utility_vector']
        expected_wealth_static = np.mean(results[job_key_static]['eval_results']['wealth_vector_list'])
        std_wealth_static = np.std(results[job_key_static]['eval_results']['wealth_vector_list'])

        static_expected_wealth_list.append(expected_wealth_static)
        static_std_wealth_list.append(std_wealth_static)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel('Expected Utility')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Re-allocations')
        ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
        ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
        ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple', label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
        ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations', alpha=overlapping_alpha)
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title(r'Expected Utility vs. $\tau$, {} Price Volatility, $a={}$'.format(hi_lo, risk_averse_a))
        plt.savefig(os.path.join(dir_path, 'utility_vs_tau_a_{}_no_uniform.pdf'.format(risk_averse_a)))
        plt.close()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel('Expected Utility')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Re-allocations')
        ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
        ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
        ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple', label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
        ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations', alpha=overlapping_alpha)
        ax1.plot(tau_list, utility_uniform_dollar_list, label='UPRA', alpha=overlapping_alpha)
        ax1.plot(tau_list, utility_uniform_liq_list, label='ULRA', alpha=overlapping_alpha)
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title(r'Expected Utility vs. $\tau$, {} Price Volatility, $a={}$'.format(hi_lo, risk_averse_a))
        plt.savefig(os.path.join(dir_path, 'utility_vs_tau_a_{}_with_uniform.pdf'.format(risk_averse_a)))
        plt.close()

        fix_tau_results = results['job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, fix_tau)]
        fig, ax = plt.subplots()
        ax.set_xlabel('Bucket Index')
        ax.set_ylabel('Proportion of Capital Allocated')
        ax.set_ylim([0, 1])
        bucket_index_list = list(range(-fix_tau, fix_tau + 1, 1))
        solved_single_vector = fix_tau_results['solved_single_vector']
        nn_portions_mean = fix_tau_results['eval_results']['nn_portions_mean']
        color_vector = ax.plot(bucket_index_list, solved_single_vector[:-1], label='OIRA')[0].get_color()
        color_nn = ax.plot(bucket_index_list, nn_portions_mean[:-1], label='ODRA (Mean Allocation)')[0].get_color()
        # ax.text(0.0, 0.5, 'Unallocated Capital of OIRA: {}'.format(solved_single_vector[-1]), color=color_vector, fontsize=12)
        # ax.text(1.0, 1.0, 'Unallocated Capital of ODRA: {}'.format(nn_portions_mean[-1]), color=color_nn, fontsize=12)
        ax.legend()
        plt.title(r'Proportional Capital Allocation, {} Price Volatility, $a={}$'.format(hi_lo, risk_averse_a))
        plt.savefig(os.path.join(dir_path, 'allocations_a_{}.pdf'.format(risk_averse_a)))
        plt.close()

        fig, ax = plt.subplots()
        ax.set_xlabel('Standard Deviation of Final Wealth')
        ax.set_ylabel('Expected Final Wealth')
        ax.scatter(std_wealth_vector_list, expected_wealth_vector_list, label='OIRA', alpha=overlapping_alpha)
        ax.scatter(std_wealth_nn_list, expected_wealth_nn_list, label='ODRA', alpha=overlapping_alpha)
        ax.scatter(std_wealth_uniform_dollar_list, expected_wealth_uniform_dollar_list, label='UPRA', alpha=overlapping_alpha)
        ax.scatter(std_wealth_uniform_liq_list, expected_wealth_uniform_liq_list, label='ULRA', alpha=overlapping_alpha)
        ax.scatter([std_wealth_static], [expected_wealth_static], label='OSA', alpha=overlapping_alpha)
        ax.legend()
        plt.title(r'Mean and Std. of Final Wealth, {} Price Volatility, $a={}$'.format(hi_lo, risk_averse_a))
        plt.savefig(os.path.join(dir_path, 'wealth_vs_std_a_{}.pdf'.format(risk_averse_a)))
        plt.close()

        nn_best_tau_idx = np.argmax(utility_nn_list)
        nn_best_tau_expected_wealth_list.append(expected_wealth_nn_list[nn_best_tau_idx])
        nn_best_tau_std_wealth_list.append(std_wealth_nn_list[nn_best_tau_idx])

        vector_best_tau_idx = np.argmax(utility_vector_list)
        vector_best_tau_expected_wealth_list.append(expected_wealth_vector_list[vector_best_tau_idx])
        vector_best_tau_std_wealth_list.append(std_wealth_vector_list[vector_best_tau_idx])

    fig, ax = plt.subplots()
    ax.set_xlabel('Standard Deviation of Final Wealth')
    ax.set_ylabel('Expected Final Wealth')
    ax.scatter(vector_best_tau_std_wealth_list, vector_best_tau_expected_wealth_list, label='OIRA', alpha=overlapping_alpha)
    ax.scatter(nn_best_tau_std_wealth_list, nn_best_tau_expected_wealth_list, label='ODRA', alpha=overlapping_alpha)
    ax.scatter(static_std_wealth_list, static_expected_wealth_list, label='OSA', alpha=overlapping_alpha)
    ax.legend()
    plt.title(r'Mean and Std. of Final Wealth, {} Price Volatility'.format(hi_lo))
    plt.savefig(os.path.join(dir_path, 'wealth_vs_std_best_tau_{}.pdf'.format(hi_lo)))
    plt.close()


def generate_X_hilo_Y_a_plot(low_exp_id, high_exp_id, output_dir_path='results/output_plots'):
    tau_list = list(range(1, 21, 1))
    static_tau = 100
    risk_a_list = [0.0, 10.0, 20.0]
    fix_tau = 20
    n_X = 2
    n_Y = len(risk_a_list)
    overlapping_alpha = 1.0

    fig_utility_no_uniform, axes_utility_no_uniform = plt.subplots(n_X, n_Y, figsize=(20, 11))
    fig_utility_with_uniform, axes_utility_with_uniform = plt.subplots(n_X, n_Y, figsize=(20, 11))
    fig_allocation, axes_allocation = plt.subplots(n_X, n_Y, figsize=(20, 11))

    for i, hilo in enumerate(['Low', 'High']):
        if hilo == 'Low':
            exp_id = low_exp_id
        else:
            exp_id = high_exp_id
        dir_path = 'results/exp_{}'.format(exp_id)
        results = np.load(os.path.join(dir_path, 'results_merge.npy'), allow_pickle=True).item()['results']
        for j, risk_averse_a in enumerate(risk_a_list):
            mean_rebalance_list = []
            utility_nn_list = []
            utility_vector_list = []
            utility_uniform_dollar_list = []
            utility_uniform_liq_list = []

            expected_wealth_nn_list = []
            expected_wealth_vector_list = []
            expected_wealth_uniform_dollar_list = []
            expected_wealth_uniform_liq_list = []

            std_wealth_nn_list = []
            std_wealth_vector_list = []
            std_wealth_uniform_dollar_list = []
            std_wealth_uniform_liq_list = []

            for tau in tau_list:
                job_key = 'job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, tau)

                eval_results = results[job_key]['eval_results']
                mean_rebalance_list.append(eval_results['mean_rebalance_count'])

                utility_nn_list.append(eval_results['expected_utility_nn'])
                utility_vector_list.append(eval_results['expected_utility_vector'])
                utility_uniform_dollar_list.append(eval_results['expected_utility_uniform_dollar'])
                utility_uniform_liq_list.append(eval_results['expected_utility_uniform_liq'])

                expected_wealth_nn_list.append(np.mean(eval_results['wealth_nn_list']))
                expected_wealth_vector_list.append(np.mean(eval_results['wealth_vector_list']))
                expected_wealth_uniform_dollar_list.append(np.mean(eval_results['wealth_uniform_dollar_list']))
                expected_wealth_uniform_liq_list.append(np.mean(eval_results['wealth_uniform_liq_list']))

                std_wealth_nn_list.append(np.std(eval_results['wealth_nn_list']))
                std_wealth_vector_list.append(np.std(eval_results['wealth_vector_list']))
                std_wealth_uniform_dollar_list.append(np.std(eval_results['wealth_uniform_dollar_list']))
                std_wealth_uniform_liq_list.append(np.std(eval_results['wealth_uniform_liq_list']))

                nn_portions_mean = eval_results['nn_portions_mean']
                solved_single_vector = results[job_key]['solved_single_vector']
                print('job_key={}'.format(job_key))
                print('nn_portions_mean={}'.format(nn_portions_mean))
                print('solved_single_vector={}'.format(solved_single_vector))

            job_key_static = 'job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, static_tau)
            utility_static = results[job_key_static]['eval_results']['expected_utility_vector']
            expected_wealth_static = np.mean(results[job_key_static]['eval_results']['wealth_vector_list'])
            std_wealth_static = np.std(results[job_key_static]['eval_results']['wealth_vector_list'])

            ax1 = axes_utility_no_uniform[i, j]
            ax1.set_xlabel(r'$\tau$')
            ax1.set_ylabel('Expected Utility')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Re-allocations')
            ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple',
                     label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
            ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations',
                     alpha=overlapping_alpha)
            handles_no_uniform, labels_no_uniform = [(a + b) for a, b in
                zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.set_title(r'{} Price Volatility, $a={}$'.format(hilo, risk_averse_a))

            ax1 = axes_utility_with_uniform[i, j]
            ax1.set_xlabel(r'$\tau$')
            ax1.set_ylabel('Expected Utility')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Re-allocations')
            ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple',
                     label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
            ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations',
                     alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_uniform_dollar_list, label='UPRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_uniform_liq_list, label='ULRA', alpha=overlapping_alpha)
            handles_with_uniform, labels_with_uniform = [(a + b) for a, b in
                zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.set_title(r'{} Price Volatility, $a={}$'.format(hilo, risk_averse_a))

            ax = axes_allocation[i, j]
            fix_tau_results = results['job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, fix_tau)]
            ax.set_xlabel('Bucket Index')
            ax.set_ylabel('Proportional Capital')
            ax.set_ylim([0, 1])
            bucket_index_list = list(range(-fix_tau, fix_tau + 1, 1))
            solved_single_vector = fix_tau_results['solved_single_vector']
            nn_portions_mean = fix_tau_results['eval_results']['nn_portions_mean']
            color_vector = ax.plot(bucket_index_list, solved_single_vector[:-1], label='OIRA')[0].get_color()
            color_nn = ax.plot(bucket_index_list, nn_portions_mean[:-1], label='ODRA (Mean Allocation)')[0].get_color()
            # ax.text(0.0, 0.5, 'Unallocated Capital of OIRA: {}'.format(solved_single_vector[-1]), color=color_vector, fontsize=12)
            # ax.text(1.0, 1.0, 'Unallocated Capital of ODRA: {}'.format(nn_portions_mean[-1]), color=color_nn, fontsize=12)
            handles_allocation, labels_allocation = ax.get_legend_handles_labels()
            ax.set_title(r'{} Price Volatility, $a={}$'.format(hilo, risk_averse_a))

    fig_utility_no_uniform.tight_layout()
    # plt.suptitle(r'Expected Utility vs. $\tau$')
    fig_utility_no_uniform.legend(handles_no_uniform, labels_no_uniform, bbox_to_anchor=[0.8, 1.05], ncols=4)
    fig_utility_no_uniform.savefig(os.path.join(output_dir_path, 'utility_vs_tau_no_uniform.pdf'), bbox_inches='tight')
    plt.close(fig_utility_no_uniform)

    fig_utility_with_uniform.tight_layout()
    # plt.suptitle(r'Expected Utility vs. $\tau$')
    fig_utility_with_uniform.legend(handles_with_uniform, labels_with_uniform, bbox_to_anchor=[0.94, 1.05], ncols=6)
    fig_utility_with_uniform.savefig(os.path.join(output_dir_path, 'utility_vs_tau_with_uniform.pdf'), bbox_inches='tight')
    plt.close(fig_utility_with_uniform)

    fig_allocation.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.suptitle(r'Proportional Capital Allocation, $\tau={}$'.format(fix_tau))
    fig_allocation.legend(handles_allocation, labels_allocation, bbox_to_anchor=[0.65, 0.955], ncols=2)
    fig_allocation.savefig(os.path.join(output_dir_path, 'allocation.pdf'), bbox_inches='tight')
    plt.close(fig_allocation)


def generate_X_hilo_Y_amplitude_plot(low_exp_id, high_exp_id, output_dir_path='results/output_plots'):
    tau_list = list(range(1, 21, 1))
    # static_tau = 100
    risk_averse_a = 10.0
    amplitude_list = [0.0, 0.00003, 0.00005]

    amplitude_desc_dict = {
        0.0: r'Constant $\lambda$',
        0.00003: r'Time-varying $\lambda$',
        0.00005: r'More Time-varying $\lambda$',
    }

    fix_tau = 20
    n_X = 2
    n_Y = len(amplitude_list)
    overlapping_alpha = 1.0

    fig_utility_no_uniform, axes_utility_no_uniform = plt.subplots(n_X, n_Y, figsize=(20, 11))
    fig_allocation, axes_allocation = plt.subplots(n_X, n_Y, figsize=(20, 11))

    for i, hilo in enumerate(['Low', 'High']):
        if hilo == 'Low':
            exp_id = low_exp_id
        else:
            exp_id = high_exp_id
        dir_path = 'results/exp_{}'.format(exp_id)
        results = np.load(os.path.join(dir_path, 'results_merge.npy'), allow_pickle=True).item()['results']
        for j, amplitude in enumerate(amplitude_list):
            amplitude_desc = amplitude_desc_dict[amplitude]

            mean_rebalance_list = []
            utility_nn_list = []
            utility_vector_list = []
            utility_uniform_dollar_list = []
            utility_uniform_liq_list = []

            expected_wealth_nn_list = []
            expected_wealth_vector_list = []
            expected_wealth_uniform_dollar_list = []
            expected_wealth_uniform_liq_list = []

            std_wealth_nn_list = []
            std_wealth_vector_list = []
            std_wealth_uniform_dollar_list = []
            std_wealth_uniform_liq_list = []

            for tau in tau_list:
                job_key = 'job_gbm_0_amplitude_{}_tau_{}'.format(amplitude, tau)

                eval_results = results[job_key]['eval_results']
                mean_rebalance_list.append(eval_results['mean_rebalance_count'])

                utility_nn_list.append(eval_results['expected_utility_nn'])
                utility_vector_list.append(eval_results['expected_utility_vector'])
                utility_uniform_dollar_list.append(eval_results['expected_utility_uniform_dollar'])
                utility_uniform_liq_list.append(eval_results['expected_utility_uniform_liq'])

                expected_wealth_nn_list.append(np.mean(eval_results['wealth_nn_list']))
                expected_wealth_vector_list.append(np.mean(eval_results['wealth_vector_list']))
                expected_wealth_uniform_dollar_list.append(np.mean(eval_results['wealth_uniform_dollar_list']))
                expected_wealth_uniform_liq_list.append(np.mean(eval_results['wealth_uniform_liq_list']))

                std_wealth_nn_list.append(np.std(eval_results['wealth_nn_list']))
                std_wealth_vector_list.append(np.std(eval_results['wealth_vector_list']))
                std_wealth_uniform_dollar_list.append(np.std(eval_results['wealth_uniform_dollar_list']))
                std_wealth_uniform_liq_list.append(np.std(eval_results['wealth_uniform_liq_list']))

                nn_portions_mean = eval_results['nn_portions_mean']
                solved_single_vector = results[job_key]['solved_single_vector']
                print('job_key={}'.format(job_key))
                print('nn_portions_mean={}'.format(nn_portions_mean))
                print('solved_single_vector={}'.format(solved_single_vector))

            # job_key_static = 'job_gbm_0_amplitude_{}_tau_{}'.format(amplitude, static_tau)
            # utility_static = results[job_key_static]['eval_results']['expected_utility_vector']
            # expected_wealth_static = np.mean(results[job_key_static]['eval_results']['wealth_vector_list'])
            # std_wealth_static = np.std(results[job_key_static]['eval_results']['wealth_vector_list'])

            ax1 = axes_utility_no_uniform[i, j]
            ax1.set_xlabel(r'$\tau$')
            ax1.set_ylabel('Expected Utility')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Re-allocations')
            ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
            # ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple', label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
            ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations',
                     alpha=overlapping_alpha)
            handles_no_uniform, labels_no_uniform = [(a + b) for a, b in
                zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.set_title(r'{} Volatility, {}'.format(hilo, amplitude_desc))

            ax = axes_allocation[i, j]
            fix_tau_results = results['job_gbm_0_amplitude_{}_tau_{}'.format(amplitude, fix_tau)]
            ax.set_xlabel('Bucket Index')
            ax.set_ylabel('Proportion of Capital Allocated')
            ax.set_ylim([0, 1])
            bucket_index_list = list(range(-fix_tau, fix_tau + 1, 1))
            solved_single_vector = fix_tau_results['solved_single_vector']
            nn_portions_mean = fix_tau_results['eval_results']['nn_portions_mean']
            color_vector = ax.plot(bucket_index_list, solved_single_vector[:-1], label='OIRA')[0].get_color()
            color_nn = ax.plot(bucket_index_list, nn_portions_mean[:-1], label='ODRA (Mean Allocation)')[0].get_color()
            # ax.text(0.0, 0.5, 'Unallocated Capital of OIRA: {}'.format(solved_single_vector[-1]), color=color_vector, fontsize=12)
            # ax.text(1.0, 1.0, 'Unallocated Capital of ODRA: {}'.format(nn_portions_mean[-1]), color=color_nn, fontsize=12)
            handles_allocation, labels_allocation = ax.get_legend_handles_labels()
            ax.set_title(r'{} Volatility, {}'.format(hilo, amplitude_desc))

    fig_utility_no_uniform.tight_layout()
    # plt.suptitle(r'Expected Utility vs. $\tau$')
    fig_utility_no_uniform.legend(handles_no_uniform, labels_no_uniform, bbox_to_anchor=[0.8, 1.05], ncols=3)
    fig_utility_no_uniform.savefig(os.path.join(output_dir_path, 'utility_vs_tau_no_uniform_adjust_amplitude.pdf'), bbox_inches='tight')
    plt.close(fig_utility_no_uniform)

    fig_allocation.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.suptitle(r'Proportional Capital Allocation, $\tau={}$'.format(fix_tau))
    fig_allocation.legend(handles_allocation, labels_allocation, bbox_to_anchor=[0.65, 0.955], ncols=2)
    fig_allocation.savefig(os.path.join(output_dir_path, 'allocation_adjust_amplitude.pdf'), bbox_inches='tight')
    plt.close(fig_allocation)


def generate_X_hilo_Y_mean_lambda_plot(low_exp_id, high_exp_id, output_dir_path='results/output_plots'):
    tau_list = list(range(1, 21, 1))
    # static_tau = 100
    risk_averse_a = 10.0
    mean_lambda_list = [0.000025, 0.00005, 0.000075]

    mean_lambda_desc_dict = {
        0.000025: r'Low Volume',
        0.00005: r'Medium Volume',
        0.000075: r'High Volume',
    }

    fix_tau = 20
    n_X = 2
    n_Y = len(mean_lambda_list)
    overlapping_alpha = 1.0

    fig_utility_no_uniform, axes_utility_no_uniform = plt.subplots(n_X, n_Y, figsize=(20, 11))
    fig_allocation, axes_allocation = plt.subplots(n_X, n_Y, figsize=(20, 11))

    for i, hilo in enumerate(['Low', 'High']):
        if hilo == 'Low':
            exp_id = low_exp_id
        else:
            exp_id = high_exp_id
        dir_path = 'results/exp_{}'.format(exp_id)
        results = np.load(os.path.join(dir_path, 'results_merge.npy'), allow_pickle=True).item()['results']
        for j, mean_lambda in enumerate(mean_lambda_list):
            mean_lambda_desc = mean_lambda_desc_dict[mean_lambda]

            mean_rebalance_list = []
            utility_nn_list = []
            utility_vector_list = []
            utility_uniform_dollar_list = []
            utility_uniform_liq_list = []

            expected_wealth_nn_list = []
            expected_wealth_vector_list = []
            expected_wealth_uniform_dollar_list = []
            expected_wealth_uniform_liq_list = []

            std_wealth_nn_list = []
            std_wealth_vector_list = []
            std_wealth_uniform_dollar_list = []
            std_wealth_uniform_liq_list = []

            for tau in tau_list:
                job_key = 'job_gbm_0_mlambda_{}_tau_{}'.format(mean_lambda, tau)

                eval_results = results[job_key]['eval_results']
                mean_rebalance_list.append(eval_results['mean_rebalance_count'])

                utility_nn_list.append(eval_results['expected_utility_nn'])
                utility_vector_list.append(eval_results['expected_utility_vector'])
                utility_uniform_dollar_list.append(eval_results['expected_utility_uniform_dollar'])
                utility_uniform_liq_list.append(eval_results['expected_utility_uniform_liq'])

                expected_wealth_nn_list.append(np.mean(eval_results['wealth_nn_list']))
                expected_wealth_vector_list.append(np.mean(eval_results['wealth_vector_list']))
                expected_wealth_uniform_dollar_list.append(np.mean(eval_results['wealth_uniform_dollar_list']))
                expected_wealth_uniform_liq_list.append(np.mean(eval_results['wealth_uniform_liq_list']))

                std_wealth_nn_list.append(np.std(eval_results['wealth_nn_list']))
                std_wealth_vector_list.append(np.std(eval_results['wealth_vector_list']))
                std_wealth_uniform_dollar_list.append(np.std(eval_results['wealth_uniform_dollar_list']))
                std_wealth_uniform_liq_list.append(np.std(eval_results['wealth_uniform_liq_list']))

                nn_portions_mean = eval_results['nn_portions_mean']
                solved_single_vector = results[job_key]['solved_single_vector']
                print('job_key={}'.format(job_key))
                print('nn_portions_mean={}'.format(nn_portions_mean))
                print('solved_single_vector={}'.format(solved_single_vector))

            ax1 = axes_utility_no_uniform[i, j]
            ax1.set_xlabel(r'$\tau$')
            ax1.set_ylabel('Expected Utility')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Re-allocations')
            ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
            # ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple', label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
            ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations',
                     alpha=overlapping_alpha)
            handles_no_uniform, labels_no_uniform = [(a + b) for a, b in
                zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.set_title(r'{} Volatility, {}'.format(hilo, mean_lambda_desc))

            ax = axes_allocation[i, j]
            fix_tau_results = results['job_gbm_0_mlambda_{}_tau_{}'.format(mean_lambda, fix_tau)]
            ax.set_xlabel('Bucket Index')
            ax.set_ylabel('Proportion of Capital Allocated')
            ax.set_ylim([0, 1])
            bucket_index_list = list(range(-fix_tau, fix_tau + 1, 1))
            solved_single_vector = fix_tau_results['solved_single_vector']
            nn_portions_mean = fix_tau_results['eval_results']['nn_portions_mean']
            color_vector = ax.plot(bucket_index_list, solved_single_vector[:-1], label='OIRA')[0].get_color()
            color_nn = ax.plot(bucket_index_list, nn_portions_mean[:-1], label='ODRA (Mean Allocation)')[0].get_color()
            # ax.text(0.0, 0.5, 'Unallocated Capital of OIRA: {}'.format(solved_single_vector[-1]), color=color_vector, fontsize=12)
            # ax.text(1.0, 1.0, 'Unallocated Capital of ODRA: {}'.format(nn_portions_mean[-1]), color=color_nn, fontsize=12)
            handles_allocation, labels_allocation = ax.get_legend_handles_labels()
            ax.set_title(r'{} Volatility, {}'.format(hilo, mean_lambda_desc))

    fig_utility_no_uniform.tight_layout()
    # plt.suptitle(r'Expected Utility vs. $\tau$')
    fig_utility_no_uniform.legend(handles_no_uniform, labels_no_uniform, bbox_to_anchor=[0.8, 1.05], ncols=3)
    fig_utility_no_uniform.savefig(os.path.join(output_dir_path, 'utility_vs_tau_no_uniform_adjust_mean_lambda.pdf'), bbox_inches='tight')
    plt.close(fig_utility_no_uniform)

    fig_allocation.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.suptitle(r'Proportional Capital Allocation, $\tau={}$'.format(fix_tau))
    fig_allocation.legend(handles_allocation, labels_allocation, bbox_to_anchor=[0.65, 0.955], ncols=2)
    fig_allocation.savefig(os.path.join(output_dir_path, 'allocation_adjust_mean_lambda.pdf'), bbox_inches='tight')
    plt.close(fig_allocation)


def generate_X_rcost_Y_a_plot(exp_id, output_dir_path='results/output_plots'):
    dir_path = 'results/exp_{}'.format(exp_id)
    tau_list = list(range(1, 21, 1))
    static_tau = 100
    risk_a_list = [0.0, 10.0, 20.0]
    fix_tau = 20
    rcost_list = [0.005, 0.01, 0.015]
    n_X = len(rcost_list)
    n_Y = len(risk_a_list)
    overlapping_alpha = 1.0

    fig_utility_no_uniform, axes_utility_no_uniform = plt.subplots(n_X, n_Y, figsize=(20, 11))
    fig_utility_with_uniform, axes_utility_with_uniform = plt.subplots(n_X, n_Y, figsize=(20, 11))
    fig_allocation, axes_allocation = plt.subplots(n_X, n_Y, figsize=(20, 11))

    for i, rcost in enumerate(rcost_list):
        results = np.load(os.path.join(dir_path, 'results_merge.npy'), allow_pickle=True).item()['results']
        for j, risk_averse_a in enumerate(risk_a_list):
            mean_rebalance_list = []
            utility_nn_list = []
            utility_vector_list = []
            utility_uniform_dollar_list = []
            utility_uniform_liq_list = []

            expected_wealth_nn_list = []
            expected_wealth_vector_list = []
            expected_wealth_uniform_dollar_list = []
            expected_wealth_uniform_liq_list = []

            std_wealth_nn_list = []
            std_wealth_vector_list = []
            std_wealth_uniform_dollar_list = []
            std_wealth_uniform_liq_list = []

            for tau in tau_list:
                job_key = 'job_gbm_0_a_{}_tau_{}_rcost_{}'.format(risk_averse_a, tau, rcost)

                eval_results = results[job_key]['eval_results']
                mean_rebalance_list.append(eval_results['mean_rebalance_count'])

                utility_nn_list.append(eval_results['expected_utility_nn'])
                utility_vector_list.append(eval_results['expected_utility_vector'])
                utility_uniform_dollar_list.append(eval_results['expected_utility_uniform_dollar'])
                utility_uniform_liq_list.append(eval_results['expected_utility_uniform_liq'])

                expected_wealth_nn_list.append(np.mean(eval_results['wealth_nn_list']))
                expected_wealth_vector_list.append(np.mean(eval_results['wealth_vector_list']))
                expected_wealth_uniform_dollar_list.append(np.mean(eval_results['wealth_uniform_dollar_list']))
                expected_wealth_uniform_liq_list.append(np.mean(eval_results['wealth_uniform_liq_list']))

                std_wealth_nn_list.append(np.std(eval_results['wealth_nn_list']))
                std_wealth_vector_list.append(np.std(eval_results['wealth_vector_list']))
                std_wealth_uniform_dollar_list.append(np.std(eval_results['wealth_uniform_dollar_list']))
                std_wealth_uniform_liq_list.append(np.std(eval_results['wealth_uniform_liq_list']))

                nn_portions_mean = eval_results['nn_portions_mean']
                solved_single_vector = results[job_key]['solved_single_vector']
                print('job_key={}'.format(job_key))
                print('nn_portions_mean={}'.format(nn_portions_mean))
                print('solved_single_vector={}'.format(solved_single_vector))

            job_key_static = 'job_gbm_0_a_{}_tau_{}_rcost_{}'.format(risk_averse_a, static_tau, rcost)
            utility_static = results[job_key_static]['eval_results']['expected_utility_vector']

            ax1 = axes_utility_no_uniform[i, j]
            ax1.set_xlabel(r'$\tau$')
            ax1.set_ylabel('Expected Utility')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Re-allocations')
            ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
            # ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple',
            #          label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
            ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations',
                     alpha=overlapping_alpha)
            handles_no_uniform, labels_no_uniform = [(a + b) for a, b in
                zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.set_title(r'$\eta={}$, $a={}$'.format(rcost, risk_averse_a))

            ax1 = axes_utility_with_uniform[i, j]
            ax1.set_xlabel(r'$\tau$')
            ax1.set_ylabel('Expected Utility')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Number of Re-allocations')
            ax1.plot(tau_list, utility_vector_list, label='OIRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_nn_list, label='ODRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, [utility_static] * len(tau_list), '--', color='tab:purple',
                     label=r'OSA ($\tau = \infty$)', alpha=overlapping_alpha)
            ax2.plot(tau_list, mean_rebalance_list, ':', color='tab:brown', label='Expected Number of Re-allocations',
                     alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_uniform_dollar_list, label='UPRA', alpha=overlapping_alpha)
            ax1.plot(tau_list, utility_uniform_liq_list, label='ULRA', alpha=overlapping_alpha)
            handles_with_uniform, labels_with_uniform = [(a + b) for a, b in
                zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            ax1.set_title(r'$\eta={}$, $a={}$'.format(rcost, risk_averse_a))

            ax = axes_allocation[i, j]
            fix_tau_results = results['job_gbm_0_a_{}_tau_{}_rcost_{}'.format(risk_averse_a, fix_tau, rcost)]
            ax.set_xlabel('Bucket Index')
            ax.set_ylabel('Proportional Capital')
            ax.set_ylim([0, 1])
            bucket_index_list = list(range(-fix_tau, fix_tau + 1, 1))
            solved_single_vector = fix_tau_results['solved_single_vector']
            nn_portions_mean = fix_tau_results['eval_results']['nn_portions_mean']
            color_vector = ax.plot(bucket_index_list, solved_single_vector[:-1], label='OIRA')[0].get_color()
            color_nn = ax.plot(bucket_index_list, nn_portions_mean[:-1], label='ODRA (Mean Allocation)')[0].get_color()
            # ax.text(0.0, 0.5, 'Unallocated Capital of OIRA: {}'.format(solved_single_vector[-1]), color=color_vector, fontsize=12)
            # ax.text(1.0, 1.0, 'Unallocated Capital of ODRA: {}'.format(nn_portions_mean[-1]), color=color_nn, fontsize=12)
            handles_allocation, labels_allocation = ax.get_legend_handles_labels()
            ax.set_title(r'$\eta={}$, $a={}$'.format(rcost, risk_averse_a))

    fig_utility_no_uniform.tight_layout()
    # plt.suptitle(r'Expected Utility vs. $\tau$')
    fig_utility_no_uniform.legend(handles_no_uniform, labels_no_uniform, bbox_to_anchor=[0.8, 1.05], ncols=3)
    fig_utility_no_uniform.savefig(os.path.join(output_dir_path, 'utility_vs_tau_no_uniform_adjust_rcost.pdf'), bbox_inches='tight')
    plt.close(fig_utility_no_uniform)

    fig_utility_with_uniform.tight_layout()
    # plt.suptitle(r'Expected Utility vs. $\tau$')
    fig_utility_with_uniform.legend(handles_with_uniform, labels_with_uniform, bbox_to_anchor=[0.9, 1.05], ncols=6)
    fig_utility_with_uniform.savefig(os.path.join(output_dir_path, 'utility_vs_tau_with_uniform_adjust_rcost.pdf'), bbox_inches='tight')
    plt.close(fig_utility_with_uniform)

    fig_allocation.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.suptitle(r'Proportional Capital Allocation, $\tau={}$'.format(fix_tau))
    fig_allocation.legend(handles_allocation, labels_allocation, bbox_to_anchor=[0.65, 0.955], ncols=2)
    fig_allocation.savefig(os.path.join(output_dir_path, 'allocation_adjust_rcost.pdf'), bbox_inches='tight')
    plt.close(fig_allocation)


def generate_risk_aversion_plot(exp_id, output_dir_path='results/output_plots'):
    dir_path = 'results/exp_{}'.format(exp_id)
    tau_list = [1, 5, 10, 15, 20]
    static_tau = 100
    risk_a_list = [2.0 * x for x in range(0, 11, 1)]
    fix_tau = 20
    rcost_list = [0.005, 0.01, 0.015]
    n_X = 3
    n_Y = 4
    overlapping_alpha = 1.0

    fig_allocation, axes_allocation = plt.subplots(n_X, n_Y, figsize=(20, 11))
    axes_allocation_list = axes_allocation.flatten()

    results = np.load(os.path.join(dir_path, 'results_merge.npy'), allow_pickle=True).item()['results']
    for i, risk_averse_a in enumerate(risk_a_list):
        mean_rebalance_list = []
        utility_nn_list = []
        utility_vector_list = []
        utility_uniform_dollar_list = []
        utility_uniform_liq_list = []

        expected_wealth_nn_list = []
        expected_wealth_vector_list = []
        expected_wealth_uniform_dollar_list = []
        expected_wealth_uniform_liq_list = []

        std_wealth_nn_list = []
        std_wealth_vector_list = []
        std_wealth_uniform_dollar_list = []
        std_wealth_uniform_liq_list = []

        for tau in tau_list:
            job_key = 'job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, tau)

            eval_results = results[job_key]['eval_results']
            mean_rebalance_list.append(eval_results['mean_rebalance_count'])

            utility_nn_list.append(eval_results['expected_utility_nn'])
            utility_vector_list.append(eval_results['expected_utility_vector'])
            utility_uniform_dollar_list.append(eval_results['expected_utility_uniform_dollar'])
            utility_uniform_liq_list.append(eval_results['expected_utility_uniform_liq'])

            expected_wealth_nn_list.append(np.mean(eval_results['wealth_nn_list']))
            expected_wealth_vector_list.append(np.mean(eval_results['wealth_vector_list']))
            expected_wealth_uniform_dollar_list.append(np.mean(eval_results['wealth_uniform_dollar_list']))
            expected_wealth_uniform_liq_list.append(np.mean(eval_results['wealth_uniform_liq_list']))

            std_wealth_nn_list.append(np.std(eval_results['wealth_nn_list']))
            std_wealth_vector_list.append(np.std(eval_results['wealth_vector_list']))
            std_wealth_uniform_dollar_list.append(np.std(eval_results['wealth_uniform_dollar_list']))
            std_wealth_uniform_liq_list.append(np.std(eval_results['wealth_uniform_liq_list']))

            nn_portions_mean = eval_results['nn_portions_mean']
            solved_single_vector = results[job_key]['solved_single_vector']
            print('job_key={}'.format(job_key))
            print('nn_portions_mean={}'.format(nn_portions_mean))
            print('solved_single_vector={}'.format(solved_single_vector))

        job_key_static = 'job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, static_tau)
        utility_static = results[job_key_static]['eval_results']['expected_utility_vector']
        expected_wealth_static = np.mean(results[job_key_static]['eval_results']['wealth_vector_list'])
        std_wealth_static = np.std(results[job_key_static]['eval_results']['wealth_vector_list'])

        ax = axes_allocation_list[i]
        fix_tau_results = results['job_gbm_0_a_{}_tau_{}'.format(risk_averse_a, fix_tau)]
        ax.set_xlabel('Bucket Index')
        ax.set_ylabel('Proportional Capital')
        ax.set_ylim([0, 1])
        bucket_index_list = list(range(-fix_tau, fix_tau + 1, 1))
        solved_single_vector = fix_tau_results['solved_single_vector']
        nn_portions_mean = fix_tau_results['eval_results']['nn_portions_mean']
        color_vector = ax.plot(bucket_index_list, solved_single_vector[:-1], label='OIRA')[0].get_color()
        color_nn = ax.plot(bucket_index_list, nn_portions_mean[:-1], label='ODRA (Mean Allocation)')[0].get_color()
        # ax.text(0.0, 0.5, 'Unallocated Capital of OIRA: {}'.format(solved_single_vector[-1]), color=color_vector, fontsize=12)
        # ax.text(1.0, 1.0, 'Unallocated Capital of ODRA: {}'.format(nn_portions_mean[-1]), color=color_nn, fontsize=12)
        handles_allocation, labels_allocation = ax.get_legend_handles_labels()
        ax.set_title(r'$a={}$'.format(risk_averse_a))

    fig_allocation.delaxes(axes_allocation_list[-1])
    fig_allocation.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.suptitle(r'Proportional Capital Allocation, $\tau={}$'.format(fix_tau))
    fig_allocation.legend(handles_allocation, labels_allocation, bbox_to_anchor=[0.65, 0.955], ncols=2)
    fig_allocation.savefig(os.path.join(output_dir_path, 'allocation_adjust_a.pdf'), bbox_inches='tight')
    plt.close(fig_allocation)


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 20})
