import math
from abc import ABC, abstractmethod
import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.special import comb
from pathos.multiprocessing import ProcessingPool


# price: price of 1 Token A in units of Token B

# delta functions

# delta change of amount of Token A
def delta_x(p1, p2):
    return 1 / math.sqrt(p2) - 1 / math.sqrt(p1)


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
    p_div = (p_high / p_low) ** (1 / (num_prices - 1))
    prices = [p_low * (p_div ** i) for i in range(num_prices)]
    return prices, p_div


# generate a binomial transition matrix
# n: number of prices
# w: width of the binomial expansion
# p: probability of a price increase ?
def binomial_matrix(n, w, p):
    # Array to hold the computed binomial expansion coefficients
    coeffs = [0 for i in range(n)]
    matrix = np.zeros((n, n))
    # Compute the coefficients given w, n, p

    for k in range(-w, w + 1):
        newC = comb(w * 2, w + k, exact=True) * (p ** (w + k)) * ((1 - p) ** (w - k))
        coeffs[k + n // 2] = newC

    # Set the desired coefficients to the location in the matrix. matrix[i][j] represents a jump from i to j
    for i in range(len(matrix)):
        for j in range(max(0, i - w), min(len(matrix), i + w + 1)):
            matrix[i][j] = coeffs[n // 2 + (j - i)]

    sum_of_rows = matrix.sum(axis=1)
    normalizedMatrix = matrix / sum_of_rows[:, np.newaxis]
    # print(normalizedMatrix)
    # plt.plot(coeffs)
    return normalizedMatrix


class PriceModel(ABC):

    @abstractmethod
    def get_price_sequence_sample(self, p0_index, t_horizon):
        pass


# sample a sequence of external prices with a transition matrix
class BinomialMatrixPriceModel(PriceModel):

    def __init__(self, transition_matrix, discrete_prices):
        self.transition_matrix = transition_matrix
        self.discrete_prices = discrete_prices

    def get_price_sequence_sample(self, p0, t_horizon):
        p0_index = self.discrete_prices.index(p0)
        num_prices = len(self.discrete_prices)
        current_index = p0_index
        price_seq = [p0]
        for t in range(t_horizon):
            next_index = np.random.choice(num_prices, 1, p=self.transition_matrix[current_index])[0]
            price_seq.append(self.discrete_prices[next_index])
            current_index = next_index
        return price_seq


class GeometricBrownianMotionPriceModel(PriceModel):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_price_sequence_sample(self, p0, t_horizon):
        price_seq = [p0]
        for t in range(t_horizon):
            price_seq.append(price_seq[-1] * np.exp(self.mu + self.sigma * np.random.normal()))
        return price_seq


# get a sequence of pool prices and corresponding external prices based on a sequence of external prices
def get_dual_price_sequence(external_prices, fee_rate, num_non_arb, non_arb_lambda):
    current_external_price = external_prices[0]
    current_pool_price = current_external_price
    pool_price_seq = [current_pool_price]
    external_price_seq = [current_external_price]
    for i in range(1, len(external_prices)):
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

        current_external_price = external_prices[i]
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


def solve_single_vector_pgd(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, num_non_arb, non_arb_lambda,
                            rebalance_cost_coeff, risk_averse_a, initial_wealth,
                            tau, exponential_value=1.0001, sgd_batch_size=10, learning_rate=1e-3,
                            max_training_steps=10000, min_delta=1e-6,
                            patience=100):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, num_non_arb, non_arb_lambda)
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
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0],
                                                                             external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            for t in range(1, len(pool_price_seq)):
                current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(
                        pool_price_seq) - 1:
                    # end of episode
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_t:t + 1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets,
                                                                                         pool_price_seq[t],
                                                                                         external_price_seq[t])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] +
                                                         earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) *
                                                        inverse_price_of_liq_list[b])
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= (1. - rebalance_cost_coeff) * torch.dot(theta, value_multiplicative_arr)

                    if t == len(pool_price_seq) - 1:
                        break

                    # start of new episode
                    rebalance_count += 1
                    episode_start_t = t
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau,
                                                                        exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t],
                                                                                     external_price_seq[t])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            utility = risk_averse_utility(wealth, risk_averse_a)
            loss += -utility
            rebalance_count_list.append(rebalance_count)

        mean_rebalance_count = np.mean(rebalance_count_list)
        # print('mean_rebalance_count: {}'.format(mean_rebalance_count))

        loss /= sgd_batch_size

        if loss < best_loss - min_delta:
            best_loss = loss
            # best_theta = theta
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        loss.backward()
        # print('theta.grad: {}'.format(theta.grad))

        with torch.no_grad():
            theta -= learning_rate * theta.grad
            theta.grad.zero_()

        # print('step: {}, loss: {}'.format(step, loss.item()))
        theta = torch.tensor(project_to_sum_le_one(theta.detach().numpy()).copy(), device=device, dtype=dtype,
                             requires_grad=True)
        # print('theta: {}'.format(theta.detach().numpy()))

    return project_to_sum_le_one(theta.detach().numpy())


def solve_single_vector(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, num_non_arb, non_arb_lambda,
                        rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                        exponential_value=1.0001, sgd_batch_size=10, learning_rate=1e-3, max_training_steps=10000,
                        min_delta=1e-6, patience=100):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate,
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
        # print('portions: {}'.format(portions.detach().numpy()))

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
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0],
                                                                             external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            for t in range(1, len(pool_price_seq)):
                current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(
                        pool_price_seq) - 1:
                    # rebalance
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_t:t + 1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets,
                                                                                         pool_price_seq[t],
                                                                                         external_price_seq[t])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] +
                                                         earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) *
                                                        inverse_price_of_liq_list[b])
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= (1. - rebalance_cost_coeff) * torch.dot(portions, value_multiplicative_arr)

                    if t == len(pool_price_seq) - 1:
                        break

                    rebalance_count += 1
                    episode_start_t = t
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau,
                                                                        exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t],
                                                                                     external_price_seq[t])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

            utility = risk_averse_utility(wealth, risk_averse_a)
            loss += -utility
            rebalance_count_list.append(rebalance_count)

        mean_rebalance_count = np.mean(rebalance_count_list)
        # print('mean_rebalance_count: {}'.format(mean_rebalance_count))

        loss /= sgd_batch_size

        if loss < best_loss - min_delta:
            best_loss = loss
            # best_theta = theta
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        optimizer.zero_grad()
        loss.backward()
        # print('theta.grad: {}'.format(theta.grad))
        optimizer.step()

        # print('step: {}, loss: {}'.format(step, loss.item()))

    return project_to_sum_le_one(theta.detach().numpy())


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


def solve_nn_policy(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, num_non_arb, non_arb_lambda,
                    rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                    max_price_range, max_bucket_range, exponential_value=1.0001, sgd_batch_size=10, learning_rate=1e-3,
                    max_training_steps=10000, min_delta=1e-6, patience=100):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, num_non_arb, non_arb_lambda)
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
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0],
                                                                             external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
            portions = model(torch.tensor(
                [0. / t_horizon, pool_price_seq[0] / max_price_range, center_bucket / max_bucket_range, wealth],
                dtype=dtype))

            if portions_sum is None:
                portions_sum = portions.clone()
            else:
                portions_sum += portions.clone()
            portions_count += 1

            for t in range(1, len(pool_price_seq)):
                current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(
                        pool_price_seq) - 1:
                    # rebalance
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_t:t + 1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets,
                                                                                         pool_price_seq[t],
                                                                                         external_price_seq[t])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] +
                                                         earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) *
                                                        inverse_price_of_liq_list[b])
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= (1. - rebalance_cost_coeff) * torch.dot(portions, value_multiplicative_arr)

                    if t == len(pool_price_seq) - 1:
                        break

                    rebalance_count += 1
                    episode_start_t = t
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau,
                                                                        exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t],
                                                                                     external_price_seq[t])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
                    portions = model(torch.tensor(
                        [t / t_horizon, pool_price_seq[t] / max_price_range, center_bucket / max_bucket_range, wealth],
                        dtype=dtype))

                    if portions_sum is None:
                        portions_sum = portions.clone()
                    else:
                        portions_sum += portions.clone()
                    portions_count += 1

            utility = risk_averse_utility(wealth, risk_averse_a)
            loss -= utility
            rebalance_count_list.append(rebalance_count)

        portions_mean = portions_sum / portions_count
        # print('portions_mean: {}'.format(portions_mean))

        mean_rebalance_count = np.mean(rebalance_count_list)
        # print('mean_rebalance_count: {}'.format(mean_rebalance_count))

        loss /= sgd_batch_size

        if loss < best_loss - min_delta:
            best_loss = loss
            # best_model = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('grad: {}'.format(model.fc1.bias.grad))

        # print('step: {}, loss: {}'.format(step, loss.item()))

    return model.state_dict()


def evaluate_strategies(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, num_non_arb, non_arb_lambda,
                        rebalance_cost_coeff, risk_averse_a, initial_wealth, tau, exponential_value, solved_nn_policy,
                        solved_single_vector):
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, num_non_arb, non_arb_lambda)
        price_seq_samples.append(dual_price_seq)

    # projected gradient descent
    dtype = torch.double
    device = torch.device("cpu")

    # input dimensions: [t / t_horizon, current_price, current_bucket, current_wealth]
    nn_model = NeuralNetworkPolicy(4, 2 * tau + 1 + 1).double()
    nn_model.load_state_dict(solved_nn_policy)

    solved_single_vector = torch.tensor(solved_single_vector, dtype=dtype)
    uniform_vector = torch.tensor([1. / (2 * tau + 1)] * (2 * tau + 1) + [0.], dtype=dtype)

    rebalance_count_list = []
    nn_portions_sum = None
    nn_portions_count = 0

    wealth_nn_list = []
    wealth_vector_list = []
    wealth_uniform_list = []
    utility_nn_list = []
    utility_vector_list = []
    utility_uniform_list = []

    for i in range(num_price_seq_samples):
        price_seq_sample = price_seq_samples[i]
        external_price_seq = price_seq_sample[0]
        pool_price_seq = price_seq_sample[1]

        # initialize
        rebalance_count = 1
        wealth_nn = initial_wealth
        wealth_vector = initial_wealth
        wealth_uniform = initial_wealth
        episode_start_t = 0
        center_bucket = find_bucket_id(pool_price_seq[0], exponential_value=exponential_value)
        active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
        price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0],
                                                                         external_price_seq[0])
        inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

        for t in range(1, len(pool_price_seq)):
            current_bucket = find_bucket_id(pool_price_seq[t], exponential_value=exponential_value)
            if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or t == len(
                    pool_price_seq) - 1:
                # rebalance
                earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                    active_buckets, pool_price_seq[episode_start_t:t + 1], fee_rate)
                new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets,
                                                                                     pool_price_seq[t],
                                                                                     external_price_seq[t])
                value_multiplicative_arr = []
                for b in range(len(active_buckets)):
                    value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[t] +
                                                     earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) *
                                                    inverse_price_of_liq_list[b])
                value_multiplicative_arr.append(1.0)  # for not allocating
                value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))

                nn_portions = nn_model(torch.tensor(
                    [t / t_horizon, pool_price_seq[t] / max_price_range, center_bucket / max_bucket_range, wealth_nn],
                    dtype=dtype))

                if nn_portions_sum is None:
                    nn_portions_sum = nn_portions.clone()
                else:
                    nn_portions_sum += nn_portions.clone()
                nn_portions_count += 1

                wealth_nn *= (1. - rebalance_cost_coeff) * torch.dot(nn_portions, value_multiplicative_arr)
                wealth_vector *= (1. - rebalance_cost_coeff) * torch.dot(solved_single_vector, value_multiplicative_arr)
                wealth_uniform *= (1. - rebalance_cost_coeff) * torch.dot(uniform_vector, value_multiplicative_arr)

                if t == len(pool_price_seq) - 1:
                    break

                rebalance_count += 1
                episode_start_t = t
                center_bucket = current_bucket
                active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau,
                                                                    exponential_value=exponential_value)
                price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[t],
                                                                                 external_price_seq[t])
                inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

        utility_nn = risk_averse_utility(wealth_nn, risk_averse_a)
        utility_vector = risk_averse_utility(wealth_vector, risk_averse_a)
        utility_uniform = risk_averse_utility(wealth_uniform, risk_averse_a)

        wealth_nn_list.append(wealth_nn.detach().numpy())
        wealth_vector_list.append(wealth_vector.detach().numpy())
        wealth_uniform_list.append(wealth_uniform.detach().numpy())

        utility_nn_list.append(utility_nn.detach().numpy())
        utility_vector_list.append(utility_vector.detach().numpy())
        utility_uniform_list.append(utility_uniform.detach().numpy())

        rebalance_count_list.append(rebalance_count)

    mean_rebalance_count = np.mean(rebalance_count_list)
    nn_portions_mean = nn_portions_sum / nn_portions_count

    expected_wealth_nn = np.mean(wealth_nn_list)
    expected_wealth_vector = np.mean(wealth_vector_list)
    expected_wealth_uniform = np.mean(wealth_uniform_list)

    expected_utility_nn = np.mean(utility_nn_list)
    expected_utility_vector = np.mean(utility_vector_list)
    expected_utility_uniform = np.mean(utility_uniform_list)

    results = {
        'mean_rebalance_count': mean_rebalance_count,
        'nn_portions_mean': nn_portions_mean,
        'expected_wealth_nn': expected_wealth_nn,
        'expected_wealth_vector': expected_wealth_vector,
        'expected_wealth_uniform': expected_wealth_uniform,
        'expected_utility_nn': expected_utility_nn,
        'expected_utility_vector': expected_utility_vector,
        'expected_utility_uniform': expected_utility_uniform,
    }

    return results


def run_experiment(args):
    (job_key, gbm_mu, gbm_sigma, p0, t_horizon, num_price_seq_samples_train, num_price_seq_samples_test, fee_rate,
     num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a, initial_wealth, exponential_value, tau,
     max_price_range, max_bucket_range, nn_hyper_params, vector_hyper_params) = args

    results = {}

    results['configs'] = {
        'job_key': job_key,
        'gbm_mu': gbm_mu,
        'gbm_sigma': gbm_sigma,
        'p0': p0,
        't_horizon': t_horizon,
        'num_price_seq_samples_train': num_price_seq_samples_train,
        'num_price_seq_samples_test': num_price_seq_samples_test,
        'fee_rate': fee_rate,
        'num_non_arb': num_non_arb,
        'non_arb_lambda': non_arb_lambda,
        'rebalance_cost_coeff': rebalance_cost_coeff,
        'risk_averse_a': risk_averse_a,
        'initial_wealth': initial_wealth,
        'exponential_value': exponential_value,
        'tau': tau,
        'max_price_range': max_price_range,
        'max_bucket_range': max_bucket_range,
        'nn_hyper_params': nn_hyper_params,
        'vector_hyper_params': vector_hyper_params,
    }

    price_model = GeometricBrownianMotionPriceModel(gbm_mu, gbm_sigma)

    nn_learning_rate = nn_hyper_params['learning_rate']
    nn_sgd_batch_size = nn_hyper_params['sgd_batch_size']
    nn_max_training_steps = nn_hyper_params['max_training_steps']
    nn_min_delta = nn_hyper_params['min_delta']
    nn_patience = nn_hyper_params['patience']

    vector_learning_rate = vector_hyper_params['learning_rate']
    vector_sgd_batch_size = vector_hyper_params['sgd_batch_size']
    vector_max_training_steps = vector_hyper_params['max_training_steps']
    vector_min_delta = vector_hyper_params['min_delta']
    vector_patience = vector_hyper_params['patience']

    solved_nn_policy = solve_nn_policy(price_model, p0, t_horizon, num_price_seq_samples_train,
                                       fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff, risk_averse_a,
                                       initial_wealth, tau,
                                       max_price_range, max_bucket_range,
                                       exponential_value=exponential_value,
                                       learning_rate=nn_learning_rate, sgd_batch_size=nn_sgd_batch_size,
                                       max_training_steps=nn_max_training_steps, min_delta=nn_min_delta,
                                       patience=nn_patience)
    results['solved_nn_policy'] = solved_nn_policy

    solved_single_vector = solve_single_vector(price_model, p0, t_horizon, num_price_seq_samples_train,
                                               fee_rate, num_non_arb, non_arb_lambda, rebalance_cost_coeff,
                                               risk_averse_a,
                                               initial_wealth, tau,
                                               exponential_value=exponential_value,
                                               learning_rate=vector_learning_rate, sgd_batch_size=vector_sgd_batch_size,
                                               max_training_steps=vector_max_training_steps, min_delta=vector_min_delta,
                                               patience=vector_patience)
    results['solved_single_vector'] = solved_single_vector

    eval_results = evaluate_strategies(price_model, p0, t_horizon, num_price_seq_samples_test, fee_rate, num_non_arb,
                                       non_arb_lambda, rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                                       exponential_value, solved_nn_policy, solved_single_vector)
    results['eval_results'] = eval_results

    print('job_key: {} finished'.format(job_key))

    return results


if __name__ == '__main__':
    '''
    # binomial matrix price model
    num_prices = 301
    p_low = 200.0
    p_high = 1000.0
    prices, p_div = get_multiplicative_prices(num_prices, p_low, p_high)
    W = 5
    # pick the value of M_p which provides the martingale property
    m = math.log(p_div)
    M_p = -math.sqrt((m ** 2 + 4) / 4 / m ** 2) + (m + 2) / 2 / m
    transition_matrix = binomial_matrix(num_prices, W, M_p)
    price_model = BinomialMatrixPriceModel(transition_matrix, prices)
    p0_index = 150
    p0 = prices[p0_index]
    '''

    gbm_params_list = [(-1.1404854857288635e-06, 0.0009126285845168537),
                       (4.8350904967723856e-08, 0.0004411197134608392)]
    risk_averse_a_list = list(np.arange(0.0, 1.0, 0.1))
    tau_list = list(range(1, 11, 1))

    # geometric brownian motion price model
    # gbm_mu, gbm_sigma = -1.1404854857288635e-06, 0.0009126285845168537

    p0 = 3000.0

    t_horizon = 20000
    num_price_seq_samples_train = 500
    num_price_seq_samples_test = 500
    fee_rate = 0.003
    num_non_arb = 4
    non_arb_lambda = 0.0001
    rebalance_cost_coeff = 0.005
    # risk_averse_a = 0.5
    initial_wealth = 1.0
    exponential_value = 1.0001 ** 60
    # tau = 10
    max_price_range = 5000.0
    max_bucket_range = find_bucket_id(max_price_range, exponential_value=exponential_value)

    nn_hyper_params = {
        'learning_rate': 1e-1,
        'sgd_batch_size': 10,
        'max_training_steps': 1000,
        'min_delta': 1e-4,
        'patience': 30,
    }

    vector_hyper_params = {
        'learning_rate': 1e-1,
        'sgd_batch_size': 10,
        'max_training_steps': 1000,
        'min_delta': 1e-4,
        'patience': 30,
    }

    job_key_list = []
    job_list = []

    for i, (gbm_mu, gbm_sigma) in enumerate(gbm_params_list):
        for risk_averse_a in risk_averse_a_list:
            for tau in tau_list:
                job_key = 'job_gbm_{}_a_{}_tau_{}'.format(i, risk_averse_a, tau)
                job_key_list.append(job_key)
                job_list.append((job_key, gbm_mu, gbm_sigma, p0, t_horizon, num_price_seq_samples_train,
                                 num_price_seq_samples_test, fee_rate, num_non_arb, non_arb_lambda,
                                 rebalance_cost_coeff, risk_averse_a, initial_wealth, exponential_value,
                                 tau, max_price_range, max_bucket_range, nn_hyper_params, vector_hyper_params))

    pool = ProcessingPool(nodes=60)
    job_output_list = pool.map(run_experiment, job_list)

    results = {
        'gbm_params_list': gbm_params_list,
        'risk_averse_a_list': risk_averse_a_list,
        'tau_list': tau_list,
        'results': {},
    }

    for job_key, job_output in zip(job_key_list, job_output_list):
        results['results'][job_key] = job_output

    # save results
    np.save('results_1.npy', results)
