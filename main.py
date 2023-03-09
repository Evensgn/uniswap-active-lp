import numpy as np
import torch
import torch.nn as nn
import matplotlib
import math
from scipy.special import comb
import scipy


# price: price of 1 Token A in units of Token B

# delta functions

# delta change of amount of Token A
def delta_x(p1, p2):
    return 1/math.sqrt(p2) - 1/math.sqrt(p1)


# delta change of amount of Token B
def delta_y(p1, p2):
    return math.sqrt(p2) - math.sqrt(p1)


# amount of Token A with 1 unit of liquidity in [a, b] bucket at price p
def bucket_assets_x(a, b, p):
    if p < a:
        # price below bucket range, assets entirely in Token A
        return delta_x(b, a)
    elif p <= b:
        # price in bucket range, assets in Token A and Token B
        return delta_x(b, p)
    else:
        # price above bucket range, assets entirely in Token B
        return 0.


# amount of Token B with 1 unit of liquidity in [a, b] bucket at price p
def bucket_assets_y(a, b, p):
    if p < a:
        # price below bucket range, assets entirely in Token A
        return 0.
    elif p <= b:
        # price in bucket range, assets in Token A and Token B
        return delta_y(a, p)
    else:
        # price above bucket range, assets entirely in Token B
        return delta_y(a, b)


# amonunt of Token A and Token B locked for l unit of liquidity in [a, b] bucket at price p
def bucket_amount_locked(a, b, p, l):
    # return token A, token B amounts
    return l * bucket_assets_x(a, b, p), l * bucket_assets_y(a, b, p)


# value of 1 unit of liquidity in [a, b] bucket at price p, converted in units of Token B
def value_of_liquidity(a, b, p):
    return p * bucket_assets_x(a, b, p) + bucket_assets_y(a, b, p)


def value_of_liquidity_with_external(a, b, pool_price, external_price):
    return external_price * bucket_assets_x(a, b, pool_price) + bucket_assets_y(a, b, pool_price)


def value_of_liquidity_with_external_for_buckets(buckets, pool_price, external_price):
    price_of_liq_list = []
    for bucket in buckets:
        a, b = bucket['p_low'], bucket['p_high']
        price_of_liq_list.append(value_of_liquidity_with_external(a, b, pool_price, external_price))
    return price_of_liq_list


# generate a sequence of multiplicative discrete prices
def get_multiplicative_prices(num_prices, p_low=0.1, p_high=10.0):
    p_div = (p_high / p_low)**(1 / (num_prices - 1))
    prices = [p_low * (p_div ** i) for i in range(num_prices)]
    return prices, p_div


# generate a binomial transition matrix
# n: number of prices
# w: width of the binomial expansion
# p: probability of a price increase ?
def binomial_matrix(n, w, p):
    #Array to hold the computed binomial expansion coefficients
    coeffs = [0 for i in range(n)]
    matrix = np.zeros((n,n))
    #Compute the coefficients given w, n, p

    for k in range(-w, w+1):
        newC = comb(w * 2, w + k, exact = True) * (p ** (w+k)) * ((1-p) ** (w-k))
        coeffs[k + n//2] = newC

    #Set the desired coefficients to the location in the matrix. matrix[i][j] represents a jump from i to j
    for i in range(len(matrix)):
        for j in range(max(0, i - w), min(len(matrix), i + w + 1)):
            matrix[i][j] = coeffs[n // 2 + (j - i)]

    sum_of_rows = matrix.sum(axis=1)
    normalizedMatrix = matrix / sum_of_rows[:, np.newaxis]
    #print(normalizedMatrix)
    #plt.plot(coeffs)
    return normalizedMatrix


# sample a sequence of external prices with a transition matrix
def get_external_price_index_sequence_sample(transition_matrix, p0_index, t_horizon=100):
    num_prices = len(transition_matrix)
    current_index = p0_index
    price_idx_seq = [current_index]
    for t in range(t_horizon):
        next_index = np.random.choice(num_prices, 1, p=transition_matrix[current_index])[0]
        price_idx_seq.append(next_index)
        current_index = next_index

    return price_idx_seq


# get a sequence of pool prices and corresponding external prices based on a sequence of external prices
def get_dual_price_sequence(external_price_indices, prices, fee_rate, num_non_arb, non_arb_lambda):
    current_external_price = prices[external_price_indices[0]]
    current_pool_price = current_external_price
    pool_price_seq = [current_pool_price]
    external_price_seq = [current_external_price]
    for i in range(1, len(external_price_indices)):
        for j in range(num_non_arb):
            if np.random.random() < 0.5:
                current_pool_price *= (1 - non_arb_lambda)
            else:
                current_pool_price /= (1 - non_arb_lambda)
            pool_price_seq.append(current_pool_price)
            external_price_seq.append(current_external_price)

            if current_pool_price < (1.0 - fee_rate) * current_external_price:
                current_pool_price = (1.0 - fee_rate) * current_external_price
            elif current_pool_price > current_external_price / (1.0 - fee_rate):
                current_pool_price = current_external_price / (1.0 - fee_rate)
            pool_price_seq.append(current_pool_price)
            external_price_seq.append(current_external_price)

        current_external_price = prices[external_price_indices[i]]
        if current_pool_price < (1.0 - fee_rate) * current_external_price:
            current_pool_price = (1.0 - fee_rate) * current_external_price
        elif current_pool_price > current_external_price / (1.0 - fee_rate):
            current_pool_price = current_external_price / (1.0 - fee_rate)
        pool_price_seq.append(current_pool_price)
        external_price_seq.append(current_external_price)
    return external_price_seq, pool_price_seq


# risk averse utility function
def risk_averse_utility(x, a):
    if a != 0.:
        return (1 - torch.exp(-a * x)) / a
    else:
        return x


# create buckets
def creat_buckets(bucket_endpoints):
    buckets = []
    for bucket_id in range(1, len(bucket_endpoints)):
        newBucket = {'id': bucket_id - 1, 'p_low': bucket_endpoints[bucket_id - 1],
                     'p_high': bucket_endpoints[bucket_id]}
        buckets.append(newBucket)
    return buckets


# transaction fee collected for a single price change by 1 unit of liquidity over [a, b]
def transaction_fee_one_step(a, b, p1, p2, fee_rate):
    # return token A, token B amounts
    if (p1 < a and p2 < a) or (p1 > b and p2 > b) or p1 == p2:
        return 0., 0.
    if p1 < p2:
        return 0., fee_rate * delta_y(max(p1, a), min(p2, b))
    else:
        return fee_rate * delta_x(min(b, p1), max(p2, a)), 0.


# calculate transaction fee for a price sequence
def transaction_fee_for_sequence(buckets, pool_price_seq, fee_rate):
    earned_token_a_each_bucket = []
    earned_token_b_each_bucket = []
    for _ in buckets:
        earned_token_a_each_bucket.append(0.)
        earned_token_b_each_bucket.append(0.)

    for i in range(len(pool_price_seq) - 1):
        for j, bucket in enumerate(buckets):
            earned_tokens = transaction_fee_one_step(bucket['p_low'], bucket['p_high'], pool_price_seq[i],
                                                     pool_price_seq[i + 1], fee_rate)
            earned_token_a_each_bucket[j] += earned_tokens[0]
            earned_token_b_each_bucket[j] += earned_tokens[1]
    return earned_token_a_each_bucket, earned_token_b_each_bucket


# project a vector x to the closest z where sum(z) <= 1 and z >= 0
def project_to_sum_le_one(x):
    d = x.size

    # firstly project x to the first quadrant
    x = np.maximum(x, np.zeros(d))

    # check if sum(x) <= 1
    if np.sum(x) <= 1.0:
        return x

    x_list = []
    for i in range(d):
        x_list.append((i, x[i]))
    x_list = sorted(x_list, key=lambda t: t[1])

    # find the correct K
    v_last = None
    for i in range(d):
        K = i + 1
        if K == 1:
            v_i = (np.sum(x) - 1) / (d - K + 1)
        else:
            v_i = (v_last - x_list[i - 1][1] / (d - K + 2)) * (d - K + 2) / (d - K + 1)
        if (i == 0 or v_i >= x_list[i - 1][1]) and (v_i < x_list[i][1]):
            break
        v_last = v_i

    z = np.zeros(d)
    for i in range(d):
        if i + 1 < K:
            z[x_list[i][0]] = 0.0
        else:
            z[x_list[i][0]] = x_list[i][1] - v_i

    return z


def find_bucket_id(price, exponential_value=1.0001):
    return math.floor(math.log(price, exponential_value))


def get_buckets_given_center_bucket_id(center_bucket_id, tau, exponential_value=1.0001):
    bucket_endpoints = []
    for i in range(center_bucket_id - tau, center_bucket_id + tau + 2):
        bucket_endpoints.append(exponential_value ** i)
    return creat_buckets(bucket_endpoints)


def solve_single_vector_pgd(prices, transition_matrix, p0_index, t_horizon, num_price_seq_samples,
                           fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                           exponential_value=1.0001,
                           sgd_batch_size=10, learning_rate=1e-3, max_training_steps=10000, min_delta=1e-6, patience=100):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_idx_seq = get_external_price_index_sequence_sample(transition_matrix, p0_index, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_idx_seq, prices, fee_rate,
                                                                     num_non_arb, non_arb_lambda)
        price_seq_samples.append(dual_price_seq)

    # projected gradient descent
    dtype = torch.double
    device = torch.device("cpu")

    initial_theta = np.random.rand(2 * tau + 1 + 1)  # leave the last one for not allocating
    theta = torch.tensor(scipy.special.softmax(initial_theta).copy(), device=device, dtype=dtype, requires_grad=True)

    best_loss = np.inf
    best_theta = None
    patience_count = 0

    for step in range(max_training_steps):
        loss = 0.

        rebalance_count_list = []
        batch_idx_list = np.random.choice(len(price_seq_samples), sgd_batch_size, replace=False)
        for idx in batch_idx_list:
            price_seq_sample = price_seq_samples[idx]
            external_price_seq = price_seq_sample[0]
            pool_price_seq = price_seq_sample[1]

            # initialize
            rebalance_count = 1
            wealth = initial_wealth
            episode_start_t = 0
            center_bucket = find_bucket_id(pool_price_seq[0], exponential_value=exponential_value)
            active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0], external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            for t in range(1, len(pool_price_seq)):
                current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(pool_price_seq) - 1:
                    # end of episode
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_t:t+1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t], external_price_seq[t])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] + earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) * inverse_price_of_liq_list[b])
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= (1. - rebalance_cost_coeff) * torch.dot(theta, value_multiplicative_arr)

                    if t == len(pool_price_seq) - 1:
                        break

                    # start of new episode
                    rebalance_count += 1
                    episode_start_t = t
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t], external_price_seq[t])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            utility = risk_averse_utility(wealth, risk_averse_a)
            loss += -utility
            rebalance_count_list.append(rebalance_count)

        mean_rebalance_count = np.mean(rebalance_count_list)
        print('mean_rebalance_count: {}'.format(mean_rebalance_count))

        loss /= sgd_batch_size

        if loss < best_loss - min_delta:
            best_loss = loss
            best_theta = theta
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        loss.backward()
        print('theta.grad: {}'.format(theta.grad))

        with torch.no_grad():
            theta -= learning_rate * theta.grad
            theta.grad.zero_()

        print('step: {}, loss: {}'.format(step, loss.item()))
        theta = torch.tensor(project_to_sum_le_one(theta.detach().numpy()).copy(), device=device, dtype=dtype,
                             requires_grad=True)
        print('theta: {}'.format(theta.detach().numpy()))

    return project_to_sum_le_one(best_theta.detach().numpy())


def solve_single_vector(prices, transition_matrix, p0_index, t_horizon, num_price_seq_samples,
                        fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                        exponential_value=1.0001,
                        sgd_batch_size=10, learning_rate=1e-3, max_training_steps=10000, min_delta=1e-6, patience=100):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_idx_seq = get_external_price_index_sequence_sample(transition_matrix, p0_index, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_idx_seq, prices, fee_rate,
                                                                     num_non_arb, non_arb_lambda)
        price_seq_samples.append(dual_price_seq)

    # projected gradient descent
    dtype = torch.double
    device = torch.device("cpu")

    initial_theta = np.random.rand(2 * tau + 1 + 1)  # leave the last one for not allocating
    theta = torch.tensor(initial_theta.copy(), device=device, dtype=dtype, requires_grad=True)

    best_loss = np.inf
    best_theta = None
    patience_count = 0

    optimizer = torch.optim.Adam([theta], lr=learning_rate)

    for step in range(max_training_steps):
        portions = torch.nn.functional.softmax(theta, dim=0)
        print('portions: {}'.format(portions.detach().numpy()))

        loss = 0.

        rebalance_count_list = []
        batch_idx_list = np.random.choice(len(price_seq_samples), sgd_batch_size, replace=False)
        for idx in batch_idx_list:
            price_seq_sample = price_seq_samples[idx]
            external_price_seq = price_seq_sample[0]
            pool_price_seq = price_seq_sample[1]

            # initialize
            rebalance_count = 1
            wealth = initial_wealth
            episode_start_t = 0
            center_bucket = find_bucket_id(pool_price_seq[0], exponential_value=exponential_value)
            active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0], external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            for t in range(1, len(pool_price_seq)):
                current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(pool_price_seq) - 1:
                    # rebalance
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_t:t+1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t], external_price_seq[t])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] + earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) * inverse_price_of_liq_list[b])
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= (1. - rebalance_cost_coeff) * torch.dot(portions, value_multiplicative_arr)

                    if t == len(pool_price_seq) - 1:
                        break

                    rebalance_count += 1
                    episode_start_t = t
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t], external_price_seq[t])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            utility = risk_averse_utility(wealth, risk_averse_a)
            loss += -utility
            rebalance_count_list.append(rebalance_count)

        mean_rebalance_count = np.mean(rebalance_count_list)
        print('mean_rebalance_count: {}'.format(mean_rebalance_count))

        loss /= sgd_batch_size

        if loss < best_loss - min_delta:
            best_loss = loss
            best_theta = theta
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        optimizer.zero_grad()
        loss.backward()
        print('theta.grad: {}'.format(theta.grad))
        optimizer.step()

        print('step: {}, loss: {}'.format(step, loss.item()))

    return project_to_sum_le_one(best_theta.detach().numpy())


class NeuralNetworkPolicy(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        x = self.softmax(x)
        return x


def solve_nn_policy(prices, transition_matrix, p0_index, t_horizon, num_price_seq_samples,
                    fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                    max_price_range, max_bucket_range,
                    exponential_value=1.0001,
                    sgd_batch_size=10, learning_rate=1e-3, max_training_steps=10000, min_delta=1e-6, patience=100):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_idx_seq = get_external_price_index_sequence_sample(transition_matrix, p0_index, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_idx_seq, prices, fee_rate,
                                                                     num_non_arb, non_arb_lambda)
        price_seq_samples.append(dual_price_seq)


    # projected gradient descent
    dtype = torch.double
    device = torch.device("cpu")

    # input dimensions: [t / t_horizon, current_price, current_bucket, current_wealth]
    model = NeuralNetworkPolicy(4, 2 * tau + 1 + 1).double()

    best_loss = np.inf
    best_model = model.state_dict()
    patience_count = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.autograd.set_detect_anomaly(True)

    for step in range(max_training_steps):
        loss = 0.

        rebalance_count_list = []
        portions_sum = None
        portions_count = 0

        batch_idx_list = np.random.choice(len(price_seq_samples), sgd_batch_size, replace=False)
        for idx in batch_idx_list:
            price_seq_sample = price_seq_samples[idx]
            external_price_seq = price_seq_sample[0]
            pool_price_seq = price_seq_sample[1]

            # initialize
            rebalance_count = 1
            wealth = initial_wealth
            episode_start_t = 0
            center_bucket = find_bucket_id(pool_price_seq[0], exponential_value=exponential_value)
            active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0], external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
            portions = model(torch.tensor([0. / t_horizon, pool_price_seq[0] / max_price_range, center_bucket / max_bucket_range, wealth], dtype=dtype))

            if portions_sum is None:
                portions_sum = portions.clone()
            else:
                portions_sum += portions.clone()
            portions_count += 1

            for t in range(1, len(pool_price_seq)):
                current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(pool_price_seq) - 1:
                    # rebalance
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_t:t+1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t], external_price_seq[t])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] + earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) * inverse_price_of_liq_list[b])
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= (1. - rebalance_cost_coeff) * torch.dot(portions, value_multiplicative_arr)

                    if t == len(pool_price_seq) - 1:
                        break

                    rebalance_count += 1
                    episode_start_t = t
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t], external_price_seq[t])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
                    portions = model(torch.tensor([t / t_horizon, pool_price_seq[t] / max_price_range, center_bucket / max_bucket_range, wealth], dtype=dtype))

                    if portions_sum is None:
                        portions_sum = portions.clone()
                    else:
                        portions_sum += portions.clone()
                    portions_count += 1

            utility = risk_averse_utility(wealth, risk_averse_a)
            loss -= utility
            rebalance_count_list.append(rebalance_count)

        portions_mean = portions_sum / portions_count
        print('portions_mean: {}'.format(portions_mean))

        mean_rebalance_count = np.mean(rebalance_count_list)
        print('mean_rebalance_count: {}'.format(mean_rebalance_count))

        loss /= sgd_batch_size

        if loss < best_loss - min_delta:
            best_loss = loss
            best_model = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('grad: {}'.format(model.fc1.bias.grad))

        print('step: {}, loss: {}'.format(step, loss.item()))

    return best_model


if __name__ == '__main__':
    num_prices = 301
    p_low = 200.0
    p_high = 1000.0
    prices, p_div = get_multiplicative_prices(num_prices, p_low, p_high)
    W = 5
    # pick the value of M_p which provides the martingale property
    m = math.log(p_div)
    M_p = -math.sqrt((m ** 2 + 4) / 4 / m ** 2) + (m + 2) / 2 / m
    transition_matrix = binomial_matrix(num_prices, W, M_p)

    p0_index = 150
    t_horizon = 5000
    num_price_seq_samples = 100  # 1000
    fee_rate = 0.003
    num_non_arb = 15  # 15
    non_arb_lambda = 0.0005  # 0.00030
    rebalance_cost_coeff = 0.
    risk_averse_a = 0.0
    initial_wealth = 1.0
    tau = 10
    max_price_range = 1000.0
    exponential_value = 1.01
    max_bucket_range = find_bucket_id(max_price_range, exponential_value=exponential_value)
    learning_rate = 1e-1
    sgd_batch_size = 100
    max_training_steps = 10000
    min_delta = 1e-6
    patience = 1000

    '''
    solved_vector = solve_nn_policy(prices, transition_matrix, p0_index, t_horizon, num_price_seq_samples,
                        fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                        max_price_range, max_bucket_range,
                        exponential_value=exponential_value,
                        learning_rate=learning_rate, sgd_batch_size=sgd_batch_size,
                        max_training_steps=max_training_steps, min_delta=min_delta, patience=patience)
    '''
    solved_vector = solve_single_vector(prices, transition_matrix, p0_index, t_horizon, num_price_seq_samples,
                                    fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a,
                                    initial_wealth, tau,
                                    exponential_value=exponential_value,
                                    learning_rate=learning_rate, sgd_batch_size=sgd_batch_size,
                                    max_training_steps=max_training_steps, min_delta=min_delta, patience=patience)

