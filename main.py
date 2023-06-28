import math
from abc import ABC, abstractmethod
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
import time
from pathos.multiprocessing import ProcessingPool
import os

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
    def __init__(self, num_prices, p_low, p_high, matrix_w):
        discrete_prices, p_div = get_multiplicative_prices(num_prices, p_low, p_high)
        self.discrete_prices = discrete_prices
        m = math.log(p_div)
        M_p = -math.sqrt((m ** 2 + 4) / 4 / m ** 2) + (m + 2) / 2 / m
        self.transition_matrix = binomial_matrix(num_prices, matrix_w, M_p)

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


class ConstantTrendPriceModel(PriceModel):
    def __init__(self, constant_factor):
        self.constant_factor = constant_factor

    def get_price_sequence_sample(self, p0, t_horizon):
        price_seq = [p0]
        for t in range(t_horizon):
            price_seq.append(price_seq[-1] * self.constant_factor)
        return price_seq


class NonArbModel(ABC):
    @abstractmethod
    def get_non_arb_params(self, t):
        pass

    def get_initial_non_arb_params(self):
        return self.get_non_arb_params(0)


class ConstantNonArbModel(NonArbModel):
    def __init__(self, num_non_arb, non_arb_lambda):
        self.num_non_arb = num_non_arb
        self.non_arb_lambda = non_arb_lambda

    def get_non_arb_params(self, t):
        return self.num_non_arb, self.non_arb_lambda


class TanhNonArbModel(NonArbModel):
    def __init__(self, num_non_arb, mean_non_arb_lambda, amplitude, t_horizon, input_multiplier):
        self.num_non_arb = num_non_arb
        self.mean_non_arb_lambda = mean_non_arb_lambda
        self.amplitude = amplitude
        self.t_horizon = t_horizon
        self.input_multiplier = input_multiplier

    def get_non_arb_params(self, t):
        t_ = t / self.t_horizon - 0.5
        num_non_arb = self.num_non_arb
        non_arb_lambda = self.mean_non_arb_lambda + self.amplitude * np.tanh(t_ * self.input_multiplier)
        return num_non_arb, non_arb_lambda


class SinNonArbModel(NonArbModel):
    def __init__(self, num_non_arb, mean_non_arb_lambda, amplitude, period):
        self.num_non_arb = num_non_arb
        self.mean_non_arb_lambda = mean_non_arb_lambda
        self.amplitude = amplitude
        self.period = period

    def get_non_arb_params(self, t):
        num_non_arb = self.num_non_arb
        non_arb_lambda = self.mean_non_arb_lambda + self.amplitude * np.sin(2 * np.pi * t / self.period)
        return num_non_arb, non_arb_lambda


class StepNonArbModel(NonArbModel):
    def __init__(self, num_non_arb, non_arb_lambda, switch_t):
        self.num_non_arb = num_non_arb
        self.non_arb_lambda = non_arb_lambda
        self.switch_t = switch_t

    def get_non_arb_params(self, t):
        if t > self.switch_t:
            return self.num_non_arb, self.non_arb_lambda
        else:
            return 0, 0


# get a sequence of pool prices and corresponding external prices based on a sequence of external prices
def get_dual_price_sequence(external_prices, fee_rate, non_arb_model):
    # print('fee_rate: ', fee_rate)
    pool_price_seq = []
    external_price_seq = []
    timestep_seq = []
    current_pool_price = None
    non_arb_lambda_seq = []
    is_non_arb_seq = []
    for i in range(len(external_prices)):
        current_external_price = external_prices[i]
        current_timestep = i
        if current_pool_price is None:
            current_pool_price = current_external_price
        else:
            if current_pool_price < (1.0 - fee_rate) * current_external_price:
                current_pool_price = (1.0 - fee_rate) * current_external_price
            elif current_pool_price > current_external_price / (1.0 - fee_rate):
                current_pool_price = current_external_price / (1.0 - fee_rate)
        pool_price_seq.append(current_pool_price)
        external_price_seq.append(current_external_price)
        timestep_seq.append(current_timestep)
        num_non_arb, non_arb_lambda = non_arb_model.get_non_arb_params(current_timestep)
        non_arb_lambda_seq.append(non_arb_lambda)
        is_non_arb_seq.append(False)
        for j in range(num_non_arb):
            if np.random.random() < 0.5:
                current_pool_price *= (1 - non_arb_lambda)
            else:
                current_pool_price /= (1 - non_arb_lambda)
            pool_price_seq.append(current_pool_price)
            external_price_seq.append(current_external_price)
            timestep_seq.append(current_timestep)
            non_arb_lambda_seq.append(non_arb_lambda)
            is_non_arb_seq.append(True)

            if current_pool_price < (1.0 - fee_rate) * current_external_price:
                current_pool_price = (1.0 - fee_rate) * current_external_price
                arb_trade = True
            elif current_pool_price > current_external_price / (1.0 - fee_rate):
                current_pool_price = current_external_price / (1.0 - fee_rate)
                arb_trade = True
            else:
                arb_trade = False
            if arb_trade:
                pool_price_seq.append(current_pool_price)
                external_price_seq.append(current_external_price)
                timestep_seq.append(current_timestep)
                non_arb_lambda_seq.append(non_arb_lambda)
                is_non_arb_seq.append(False)

    return external_price_seq, pool_price_seq, timestep_seq, non_arb_lambda_seq, is_non_arb_seq


def raw_risk_averse_utility(a, x):
    if a != 0.:
        u = (1 - torch.exp(-a * x)) / a
    else:
        u = x
    return u


class RiskAverseFunction:
    def __init__(self, a):
        self.a = a
        scalar_1 = torch.tensor(1., dtype=torch.float64)
        scalar_1_1 = torch.tensor(1.1, dtype=torch.float64)
        self.utility_output_divisor = raw_risk_averse_utility(self.a, scalar_1_1) - raw_risk_averse_utility(self.a, scalar_1)
        self.base_raw_utility = raw_risk_averse_utility(self.a, scalar_1)
        self.utility_output_multiplier = 1. / self.utility_output_divisor

    def raw_risk_averse_utility(self, x):
        return raw_risk_averse_utility(self.a, x)

    def risk_averse_utility(self, x):
        return (raw_risk_averse_utility(self.a, x) - self.base_raw_utility) * self.utility_output_multiplier


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


def find_bucket_id(price, exponential_value=1.0001):
    return math.floor(math.log(price, exponential_value))


def get_buckets_given_center_bucket_id(center_bucket_id, tau, exponential_value=1.0001):
    bucket_endpoints = []
    for i in range(center_bucket_id - tau, center_bucket_id + tau + 2):
        bucket_endpoints.append(exponential_value ** i)
    return creat_buckets(bucket_endpoints)


# trading volume for a single price change in token B value for one unit of liquidity over the entire price range
def unit_volume_one_step(p1, p2):
    return delta_y(min(p1, p2), max(p1, p2))


def estimate_average_ewma_volume(price_seq_samples, only_non_arb_volume):
    volume_count = 0.0
    volume_sum = 0.0
    for price_seq_sample in price_seq_samples:
        pool_price_seq = price_seq_sample[1]
        timestep_seq = price_seq_sample[2]
        is_non_arb_seq = price_seq_sample[4]
        t = 0
        cumulative_volume = 0.0
        for i in range(1, len(pool_price_seq)):
            if is_non_arb_seq[i] or (not only_non_arb_volume):
                volume = unit_volume_one_step(pool_price_seq[i - 1], pool_price_seq[i])
                cumulative_volume += volume
            if timestep_seq[i] != t:
                volume_sum += cumulative_volume
                volume_count += 1.0
                t = timestep_seq[i]
                cumulative_volume = 0.0
    average_ewma_volume = volume_sum / volume_count
    return average_ewma_volume


def get_initial_ewma_volume(price_model, non_arb_model, p0, fee_rate, only_non_arb_volume):
    initial_num_non_arb, initial_non_arb_lambda = non_arb_model.get_initial_non_arb_params()
    initial_non_arb_model = ConstantNonArbModel(initial_num_non_arb, initial_non_arb_lambda)
    price_seq_samples = []
    for _ in range(100):
        external_price_seq = price_model.get_price_sequence_sample(p0, 100)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, initial_non_arb_model)
        price_seq_samples.append(dual_price_seq)
    initial_ewma_volume = estimate_average_ewma_volume(price_seq_samples, only_non_arb_volume)
    return initial_ewma_volume


def get_ewma_volume_seq(price_seq_sample, initial_ewma_volume, ewma_alpha, only_non_arb_volume):
    pool_price_seq = price_seq_sample[1]
    timestep_seq = price_seq_sample[2]
    is_non_arb_seq = price_seq_sample[4]

    t = 0
    cumulative_volume = 0.0
    ewma_volume = initial_ewma_volume
    ewma_volume_seq = [ewma_volume]
    for i in range(1, len(pool_price_seq)):
        if is_non_arb_seq[i] or (not only_non_arb_volume):
            volume = unit_volume_one_step(pool_price_seq[i - 1], pool_price_seq[i])
            cumulative_volume += volume
        if timestep_seq[i] != t:
            ewma_volume = ewma_volume * (1 - ewma_alpha) + cumulative_volume * ewma_alpha
            t = timestep_seq[i]
            cumulative_volume = 0.0
        ewma_volume_seq.append(ewma_volume)
    return ewma_volume_seq


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


def solve_single_vector(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, non_arb_model,
                        rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                        exponential_value=1.0001, sgd_batch_size=10, learning_rate=1e-3, max_training_steps=10000,
                        min_delta=1e-6, patience=100, projected_gd=False):
    risk_averse_function = RiskAverseFunction(risk_averse_a)
    # generate price sequences
    # print('generating price sequences...')
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, non_arb_model)

        price_seq_samples.append(dual_price_seq)

    dtype = torch.double
    device = torch.device("cpu")

    initial_theta = np.random.rand(2 * tau + 1 + 1)  # leave the last one for not allocating
    if projected_gd:
        theta = torch.tensor(scipy.special.softmax(initial_theta).copy(), device=device, dtype=dtype,
                             requires_grad=True)
    else:
        theta = torch.tensor(initial_theta.copy(), device=device, dtype=dtype, requires_grad=True)

    best_loss = np.inf
    patience_count = 0

    if not projected_gd:
        optimizer = torch.optim.Adam([theta], lr=learning_rate)

    loss_log = []

    for step in range(max_training_steps):
        portions = torch.nn.functional.softmax(theta, dim=0)

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
                                                        inverse_price_of_liq_list[b] * (1. - rebalance_cost_coeff))
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= torch.dot(portions, value_multiplicative_arr)

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

            utility = risk_averse_function.risk_averse_utility(wealth)
            loss += -utility
            rebalance_count_list.append(rebalance_count)

        loss /= sgd_batch_size

        loss_log.append(loss.item())

        if loss < best_loss - min_delta:
            best_loss = loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        if projected_gd:
            loss.backward()

            with torch.no_grad():
                theta -= learning_rate * theta.grad
                theta.grad.zero_()

            theta = torch.tensor(project_to_sum_le_one(theta.detach().numpy()).copy(), device=device, dtype=dtype,
                                 requires_grad=True)
        else:
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    if projected_gd:
        return project_to_sum_le_one(theta.detach().numpy()), loss_log
    else:
        return torch.nn.functional.softmax(theta, dim=0).detach().numpy(), loss_log


class NeuralNetworkPolicy(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        x = F.softmax(x, dim=-1)
        return x


def solve_nn_policy(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, non_arb_model,
                    rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                    price_normalization_factor, bucket_normalization_factor, wealth_normalization_factor,
                    ewma_volume_normalization_factor, exponential_value=1.0001,
                    nn_hidden_dims=(32,32),
                    sgd_batch_size=10, learning_rate=1e-3,
                    max_training_steps=10000, min_delta=1e-6, patience=100, ewma_alpha=0.2,
                    only_non_arb_ewma_volume=False):
    risk_averse_function = RiskAverseFunction(risk_averse_a)

    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, non_arb_model)
        price_seq_samples.append(dual_price_seq)

    initial_ewma_volume = get_initial_ewma_volume(price_model, non_arb_model, p0, fee_rate, only_non_arb_ewma_volume)

    ewma_volume_seqs = []
    for price_seq_sample in price_seq_samples:
        ewma_volume_seqs.append(get_ewma_volume_seq(price_seq_sample, initial_ewma_volume, ewma_alpha,
                                                    only_non_arb_ewma_volume))

    dtype = torch.double

    # input dimensions: [t / t_horizon, current_price, current_bucket, current_wealth, ewma_volume]
    model = NeuralNetworkPolicy(5, 2 * tau + 1 + 1, nn_hidden_dims).double()

    best_loss = np.inf
    patience_count = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.autograd.set_detect_anomaly(True)

    loss_log = []

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
            timestep_seq = price_seq_sample[2]
            ewma_volume_seq = ewma_volume_seqs[idx]

            # initialize
            t = 0
            rebalance_count = 1
            wealth = initial_wealth
            episode_start_i = 0
            center_bucket = find_bucket_id(pool_price_seq[0], exponential_value=exponential_value)
            active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
            price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0],
                                                                             external_price_seq[0])
            inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
            ewma_volume = ewma_volume_seq[0]

            nn_input = torch.tensor([t / t_horizon, ewma_volume / ewma_volume_normalization_factor,
                                     pool_price_seq[0] / price_normalization_factor,
                                     center_bucket / bucket_normalization_factor,
                                     wealth / wealth_normalization_factor], dtype=dtype)

            portions = model(nn_input)

            if portions_sum is None:
                portions_sum = portions.clone()
            else:
                portions_sum += portions.clone()
            portions_count += 1

            for i in range(1, len(pool_price_seq)):
                t = timestep_seq[i]
                ewma_volume = ewma_volume_seq[i]

                current_bucket = find_bucket_id(pool_price_seq[i], exponential_value=exponential_value)
                if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or i == len(
                        pool_price_seq) - 1:
                    # rebalance
                    earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                        active_buckets, pool_price_seq[episode_start_i:i + 1], fee_rate)
                    new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets,
                                                                                         pool_price_seq[i],
                                                                                         external_price_seq[i])
                    value_multiplicative_arr = []
                    for b in range(len(active_buckets)):
                        value_multiplicative_arr.append((earned_token_a_each_bucket[b] * external_price_seq[i] +
                                                         earned_token_b_each_bucket[b] + new_price_of_liq_list[b]) *
                                                        inverse_price_of_liq_list[b] * (1. - rebalance_cost_coeff))
                    value_multiplicative_arr.append(1.0)  # for not allocating
                    value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))
                    wealth *= torch.dot(portions, value_multiplicative_arr)

                    if i == len(pool_price_seq) - 1:
                        break

                    rebalance_count += 1
                    episode_start_i = i
                    center_bucket = current_bucket
                    active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau,
                                                                        exponential_value=exponential_value)
                    price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[i],
                                                                                     external_price_seq[i])
                    inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))

                    nn_input = torch.tensor([t / t_horizon, ewma_volume / ewma_volume_normalization_factor,
                                             pool_price_seq[i] / price_normalization_factor,
                                             center_bucket / bucket_normalization_factor,
                                             wealth / wealth_normalization_factor], dtype=dtype)

                    portions = model(nn_input)

                    if portions_sum is None:
                        portions_sum = portions.clone()
                    else:
                        portions_sum += portions.clone()
                    portions_count += 1

            utility = risk_averse_function.risk_averse_utility(wealth)
            loss -= utility
            rebalance_count_list.append(rebalance_count)

        loss /= sgd_batch_size

        loss_log.append(loss.item())

        if loss < best_loss - min_delta:
            best_loss = loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.state_dict(), loss_log


def evaluate_strategies(price_model, p0, t_horizon, num_price_seq_samples, fee_rate, non_arb_model,
                        rebalance_cost_coeff, risk_averse_a, initial_wealth, tau, price_normalization_factor,
                        bucket_normalization_factor, wealth_normalization_factor,
                        ewma_volume_normalization_factor, ewma_alpha,
                        only_non_arb_ewma_volume, exponential_value, nn_hidden_dims, solved_nn_policy,
                        solved_single_vector):
    risk_averse_function = RiskAverseFunction(risk_averse_a)
    # generate price sequences
    price_seq_samples = []
    for _ in range(num_price_seq_samples):
        external_price_seq = price_model.get_price_sequence_sample(p0, t_horizon)
        dual_price_seq = get_dual_price_sequence(external_price_seq, fee_rate, non_arb_model)
        price_seq_samples.append(dual_price_seq)

    initial_ewma_volume = get_initial_ewma_volume(price_model, non_arb_model, p0, fee_rate, only_non_arb_ewma_volume)

    ewma_volume_seqs = []
    for price_seq_sample in price_seq_samples:
        ewma_volume_seqs.append(get_ewma_volume_seq(price_seq_sample, initial_ewma_volume, ewma_alpha,
                                                    only_non_arb_ewma_volume))

    dtype = torch.double

    # input dimensions: [t / t_horizon, current_price, current_bucket, current_wealth, ewma_volume]
    nn_model = NeuralNetworkPolicy(5, 2 * tau + 1 + 1, nn_hidden_dims).double()
    nn_model.load_state_dict(solved_nn_policy)

    solved_single_vector = torch.tensor(solved_single_vector, dtype=dtype)

    # uniform in dollar value
    uniform_dollar_vector = torch.tensor([1. / (2 * tau + 1)] * (2 * tau + 1) + [0.], dtype=dtype)

    rebalance_count_list = []
    nn_portions_sum = None
    nn_portions_count = 0
    uniform_liq_vector_sum = None
    uniform_liq_vector_count = 0

    wealth_nn_list = []
    wealth_vector_list = []
    wealth_uniform_dollar_list = []
    wealth_uniform_liq_list = []
    utility_nn_list = []
    utility_vector_list = []
    utility_uniform_dollar_list = []
    utility_uniform_liq_list = []

    for idx in range(num_price_seq_samples):
        price_seq_sample = price_seq_samples[idx]
        external_price_seq = price_seq_sample[0]
        pool_price_seq = price_seq_sample[1]
        timestep_seq = price_seq_sample[2]
        ewma_volume_seq = ewma_volume_seqs[idx]

        # initialize
        t = 0
        ewma_volume = ewma_volume_seq[0]
        rebalance_count = 1
        wealth_nn = initial_wealth
        wealth_vector = initial_wealth
        wealth_uniform_dollar = initial_wealth
        wealth_uniform_liq = initial_wealth
        episode_start_i = 0
        center_bucket = find_bucket_id(pool_price_seq[0], exponential_value=exponential_value)
        active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau, exponential_value=exponential_value)
        price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[0],
                                                                         external_price_seq[0])
        inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
        uniform_liq_vector = torch.tensor(list(price_of_liq_list / np.sum(price_of_liq_list)) + [0.], dtype=dtype)

        if uniform_liq_vector_sum is None:
            uniform_liq_vector_sum = uniform_liq_vector
        else:
            uniform_liq_vector_sum += uniform_liq_vector
        uniform_liq_vector_count += 1

        nn_input = torch.tensor([t / t_horizon, ewma_volume / ewma_volume_normalization_factor,
                                 pool_price_seq[0] / price_normalization_factor,
                                 center_bucket / bucket_normalization_factor,
                                 wealth_nn / wealth_normalization_factor], dtype=dtype)
        nn_portions = nn_model(nn_input)

        if nn_portions_sum is None:
            nn_portions_sum = nn_portions.detach()
        else:
            nn_portions_sum += nn_portions.detach()
        nn_portions_count += 1

        for i in range(1, len(pool_price_seq)):
            t = timestep_seq[i]
            ewma_volume = ewma_volume_seq[i]

            current_bucket = find_bucket_id(pool_price_seq[i], exponential_value=exponential_value)
            if current_bucket < center_bucket - tau or current_bucket > center_bucket + tau or i == len(
                    pool_price_seq) - 1:
                # rebalance
                earned_token_a_each_bucket, earned_token_b_each_bucket = transaction_fee_for_sequence(
                    active_buckets, pool_price_seq[episode_start_i:i + 1], fee_rate)
                new_price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets,
                                                                                     pool_price_seq[i],
                                                                                     external_price_seq[i])
                value_multiplicative_arr = []
                for b in range(len(active_buckets)):
                    unit_liq_value = earned_token_a_each_bucket[b] * external_price_seq[i] + \
                                     earned_token_b_each_bucket[b] + new_price_of_liq_list[b]
                    value_multiplicative_arr.append(
                        unit_liq_value * inverse_price_of_liq_list[b] * (1. - rebalance_cost_coeff))

                value_multiplicative_arr.append(1.0)  # for not allocating
                value_multiplicative_arr = torch.from_numpy(np.array(value_multiplicative_arr))

                wealth_nn *= torch.dot(nn_portions, value_multiplicative_arr)
                wealth_vector *= torch.dot(solved_single_vector, value_multiplicative_arr)
                wealth_uniform_dollar *= torch.dot(uniform_dollar_vector, value_multiplicative_arr)
                wealth_uniform_liq *= torch.dot(uniform_liq_vector, value_multiplicative_arr)

                if i == len(pool_price_seq) - 1:
                    break

                rebalance_count += 1
                episode_start_i = i
                center_bucket = current_bucket
                active_buckets = get_buckets_given_center_bucket_id(center_bucket, tau,
                                                                    exponential_value=exponential_value)
                price_of_liq_list = value_of_liquidity_with_external_for_buckets(active_buckets, pool_price_seq[i],
                                                                                 external_price_seq[i])
                inverse_price_of_liq_list = torch.from_numpy(np.array([1. / p for p in price_of_liq_list]))
                uniform_liq_vector = torch.tensor(list(price_of_liq_list / np.sum(price_of_liq_list)) + [0.],
                                                  dtype=dtype)
                if uniform_liq_vector_sum is None:
                    uniform_liq_vector_sum = uniform_liq_vector
                else:
                    uniform_liq_vector_sum += uniform_liq_vector
                uniform_liq_vector_count += 1

                nn_input = torch.tensor([t / t_horizon, ewma_volume / ewma_volume_normalization_factor,
                                         pool_price_seq[i] / price_normalization_factor,
                                         center_bucket / bucket_normalization_factor,
                                         wealth_nn / wealth_normalization_factor], dtype=dtype)
                nn_portions = nn_model(nn_input)

                if nn_portions_sum is None:
                    nn_portions_sum = nn_portions.detach()
                else:
                    nn_portions_sum += nn_portions.detach()
                nn_portions_count += 1

        rebalance_count_list.append(rebalance_count)

        utility_nn = risk_averse_function.raw_risk_averse_utility(wealth_nn)
        utility_vector = risk_averse_function.raw_risk_averse_utility(wealth_vector)
        utility_uniform_dollar = risk_averse_function.raw_risk_averse_utility(wealth_uniform_dollar)
        utility_uniform_liq = risk_averse_function.raw_risk_averse_utility(wealth_uniform_liq)

        wealth_nn_list.append(wealth_nn.detach().numpy())
        wealth_vector_list.append(wealth_vector.detach().numpy())
        wealth_uniform_dollar_list.append(wealth_uniform_dollar.detach().numpy())
        wealth_uniform_liq_list.append(wealth_uniform_liq.detach().numpy())

        utility_nn_list.append(utility_nn.detach().numpy())
        utility_vector_list.append(utility_vector.detach().numpy())
        utility_uniform_dollar_list.append(utility_uniform_dollar.detach().numpy())
        utility_uniform_liq_list.append(utility_uniform_liq.detach().numpy())

    mean_rebalance_count = np.mean(rebalance_count_list)
    nn_portions_mean = nn_portions_sum / nn_portions_count
    uniform_liq_vector_mean = uniform_liq_vector_sum / uniform_liq_vector_count

    expected_wealth_nn = np.mean(wealth_nn_list)
    expected_wealth_vector = np.mean(wealth_vector_list)
    expected_wealth_uniform_dollar = np.mean(wealth_uniform_dollar_list)
    expected_wealth_uniform_liq = np.mean(wealth_uniform_liq_list)

    expected_utility_nn = np.mean(utility_nn_list)
    expected_utility_vector = np.mean(utility_vector_list)
    expected_utility_uniform_dollar = np.mean(utility_uniform_dollar_list)
    expected_utility_uniform_liq = np.mean(utility_uniform_liq_list)

    results = {
        'mean_rebalance_count': mean_rebalance_count,
        'nn_portions_mean': nn_portions_mean,
        'uniform_liq_vector_mean': uniform_liq_vector_mean,

        'expected_wealth_nn': expected_wealth_nn,
        'expected_wealth_vector': expected_wealth_vector,
        'expected_wealth_uniform_dollar': expected_wealth_uniform_dollar,
        'expected_wealth_uniform_liq': expected_wealth_uniform_liq,
        'expected_utility_nn': expected_utility_nn,
        'expected_utility_vector': expected_utility_vector,
        'expected_utility_uniform_dollar': expected_utility_uniform_dollar,
        'expected_utility_uniform_liq': expected_utility_uniform_liq,

        'rebalance_count_list': rebalance_count_list,
        'wealth_nn_list': wealth_nn_list,
        'wealth_vector_list': wealth_vector_list,
        'wealth_uniform_dollar_list': wealth_uniform_dollar_list,
        'wealth_uniform_liq_list': wealth_uniform_liq_list,
        'utility_nn_list': utility_nn_list,
        'utility_vector_list': utility_vector_list,
        'utility_uniform_dollar_list': utility_uniform_dollar_list,
        'utility_uniform_liq_list': utility_uniform_liq_list,
    }

    return results


def run_experiment(args):
    (dir_path, job_key, price_model_params, p0, t_horizon, num_price_seq_samples_train, num_price_seq_samples_test,
     fee_rate, non_arb_params, rebalance_cost_coeff, risk_averse_a, initial_wealth, exponential_value, tau,
     price_normalization_factor, bucket_normalization_factor, wealth_normalization_factor,
     ewma_volume_normalization_factor, ewma_alpha,
     only_non_arb_ewma_volume, nn_hyper_params, vector_hyper_params) = args

    results = {}

    results['configs'] = {
        'dir_path': dir_path,
        'job_key': job_key,
        'price_model_params': price_model_params,
        'p0': p0,
        't_horizon': t_horizon,
        'num_price_seq_samples_train': num_price_seq_samples_train,
        'num_price_seq_samples_test': num_price_seq_samples_test,
        'fee_rate': fee_rate,
        'non_arb_params': non_arb_params,
        'rebalance_cost_coeff': rebalance_cost_coeff,
        'risk_averse_a': risk_averse_a,
        'initial_wealth': initial_wealth,
        'exponential_value': exponential_value,
        'tau': tau,
        'price_normalization_factor': price_normalization_factor,
        'bucket_normalization_factor': bucket_normalization_factor,
        'ewma_volume_normalization_factor': ewma_volume_normalization_factor,
        'ewma_alpha': ewma_alpha,
        'only_non_arb_ewma_volume': only_non_arb_ewma_volume,
        'nn_hyper_params': nn_hyper_params,
        'vector_hyper_params': vector_hyper_params,
    }

    print('start job_key: {}'.format(job_key))

    price_model_type = price_model_params['type']
    if price_model_type == 'gbm':
        gbm_mu = price_model_params['gbm_mu']
        gbm_sigma = price_model_params['gbm_sigma']
        price_model = GeometricBrownianMotionPriceModel(gbm_mu, gbm_sigma)
    elif price_model_type == 'matrix':
        num_prices = price_model_params['num_prices']
        p_low = price_model_params['p_low']
        p_high = price_model_params['p_high']
        matrix_w = price_model_params['matrix_w']
        price_model = BinomialMatrixPriceModel(num_prices, p_low, p_high, matrix_w)
    elif price_model_type == 'constant-trend':
        constant_factor = price_model_params['constant_factor']
        price_model = ConstantTrendPriceModel(constant_factor)
    else:
        raise ValueError('unknown price_model_type: {}'.format(price_model_type))

    non_arb_type = non_arb_params['type']
    if non_arb_type == 'sin':
        num_non_arb = non_arb_params['num_non_arb']
        mean_non_arb_lambda = non_arb_params['mean_non_arb_lambda']
        amplitude = non_arb_params['amplitude']
        period = non_arb_params['period']
        non_arb_model = SinNonArbModel(num_non_arb, mean_non_arb_lambda, amplitude, period)
    elif non_arb_type == 'step':
        num_non_arb = non_arb_params['num_non_arb']
        non_arb_lambda = non_arb_params['non_arb_lambda']
        switch_t = non_arb_params['switch_t']
        non_arb_model = StepNonArbModel(num_non_arb, non_arb_lambda, switch_t)
    elif non_arb_type == 'tanh':
        num_non_arb = non_arb_params['num_non_arb']
        mean_non_arb_lambda = non_arb_params['mean_non_arb_lambda']
        amplitude = non_arb_params['amplitude']
        t_horizon = non_arb_params['t_horizon']
        input_multiplier = non_arb_params['input_multiplier']
        non_arb_model = TanhNonArbModel(num_non_arb, mean_non_arb_lambda, amplitude, t_horizon, input_multiplier)
    elif non_arb_type == 'constant':
        num_non_arb = non_arb_params['num_non_arb']
        non_arb_lambda = non_arb_params['non_arb_lambda']
        non_arb_model = ConstantNonArbModel(num_non_arb, non_arb_lambda)
    else:
        raise ValueError('unknown non_arb_type: {}'.format(non_arb_type))

    nn_hidden_dims = nn_hyper_params['hidden_dims']
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

    solved_nn_policy, nn_loss_log = solve_nn_policy(price_model, p0, t_horizon, num_price_seq_samples_train,
                                       fee_rate, non_arb_model, rebalance_cost_coeff, risk_averse_a,
                                       initial_wealth, tau,
                                       price_normalization_factor, bucket_normalization_factor,
                                       wealth_normalization_factor,
                                       ewma_volume_normalization_factor,
                                       exponential_value=exponential_value, nn_hidden_dims=nn_hidden_dims,
                                       learning_rate=nn_learning_rate, sgd_batch_size=nn_sgd_batch_size,
                                       max_training_steps=nn_max_training_steps, min_delta=nn_min_delta,
                                       patience=nn_patience, ewma_alpha=ewma_alpha,
                                       only_non_arb_ewma_volume=only_non_arb_ewma_volume)
    results['solved_nn_policy'] = solved_nn_policy
    results['nn_loss_log'] = nn_loss_log

    solved_single_vector, vector_loss_log = solve_single_vector(price_model, p0, t_horizon, num_price_seq_samples_train,
                                                                fee_rate, non_arb_model,
                                                                rebalance_cost_coeff,
                                                                risk_averse_a,
                                                                initial_wealth, tau,
                                                                exponential_value=exponential_value,
                                                                learning_rate=vector_learning_rate,
                                                                sgd_batch_size=vector_sgd_batch_size,
                                                                max_training_steps=vector_max_training_steps,
                                                                min_delta=vector_min_delta,
                                                                patience=vector_patience)
    results['solved_single_vector'] = solved_single_vector
    results['vector_loss_log'] = vector_loss_log

    eval_results = evaluate_strategies(price_model, p0, t_horizon, num_price_seq_samples_test, fee_rate,
                                      non_arb_model,
                                      rebalance_cost_coeff, risk_averse_a, initial_wealth, tau,
                                      price_normalization_factor, bucket_normalization_factor,
                                      wealth_normalization_factor,
                                      ewma_volume_normalization_factor, ewma_alpha, only_non_arb_ewma_volume,
                                      exponential_value, nn_hidden_dims, solved_nn_policy, solved_single_vector)

    results['eval_results'] = eval_results

    print('job_key: {} finished'.format(job_key))

    np.save(os.path.join(dir_path, '{}.npy'.format(job_key)), results)

    return results


if __name__ == '__main__':
    np.random.seed(0)
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

    # [ETH/USDT (high volatility), ETH/BTC (low volatility)]
    gbm_params_list = [(-1.1404854857288635e-06, 0.0009126285845168537),
                       (4.8350904967723856e-08, 0.0004411197134608392)]

    tau_list = list(range(1, 21, 1)) + [100]
    risk_averse_a = 10.0
    non_arb_tanh_amplitude_list = [0.0, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005]
    # different parameter value lists to choose from:
    # mean_non_arb_lambda_list = [0.000025, 0.00005, 0.000075, 0.0001]
    # risk_averse_a_list = [0.0, 10.0, 20.0]
    # rebalance_cost_coeff_list = [0.005 * x for x in range(0, 6, 1)]
    # tau_list = [1, 5, 10, 15, 20] + [100]
    # risk_averse_a_list = [2.0 * x for x in range(0, 11, 1)]

    p0 = 1.0

    t_horizon = 1000
    num_price_seq_samples_train = 1000
    num_price_seq_samples_test = 1000

    fee_rate, tick_spacing = (0.003, 10)
    exponential_value = 1.0001 ** tick_spacing

    non_arb_params = {
        'type': 'tanh',
        'num_non_arb': 10,
        'mean_non_arb_lambda': 0.00005,
        'amplitude': 0.00005,
        't_horizon': t_horizon,
        'input_multiplier': 10.0,
    }

    rebalance_cost_coeff = 0.01
    initial_wealth = 1.0
    price_normalization_factor = 1.0
    bucket_normalization_factor = 100
    wealth_normalization_factor = 2.0

    ewma_volume_normalization_factor = 0.001

    ewma_alpha = 0.1
    only_non_arb_ewma_volume = True

    nn_hyper_params = {
        'hidden_dims': (16, 16, 16, 16, 16),
        'learning_rate': 1e-3,
        'sgd_batch_size': 1,
        'max_training_steps': 10000,
        'min_delta': 1e-8,
        'patience': 10000,
    }

    vector_hyper_params = {
        'learning_rate': 1e-2,
        'sgd_batch_size': 1,
        'max_training_steps': 10000,
        'min_delta': 1e-8,
        'patience': 10000,
    }

    dir_path = os.path.join('results', 'exp_103')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    job_key_list = []
    job_list = []

    for i, (gbm_mu, gbm_sigma) in enumerate(gbm_params_list):
        # change the look based on the varying parameter
        for non_arb_tanh_amplitude in non_arb_tanh_amplitude_list:
            for tau in tau_list:
                job_key = 'job_gbm_{}_amplitude_{}_tau_{}'.format(i, non_arb_tanh_amplitude, tau)
                job_key_list.append(job_key)
                price_model_params = {
                    'type': 'gbm',
                    'gbm_mu': gbm_mu,
                    'gbm_sigma': gbm_sigma,
                }
                job_list.append((dir_path, job_key, price_model_params, p0, t_horizon, num_price_seq_samples_train,
                                 num_price_seq_samples_test, fee_rate, non_arb_params, rebalance_cost_coeff,
                                 risk_averse_a, initial_wealth, exponential_value, tau, price_normalization_factor,
                                 bucket_normalization_factor, wealth_normalization_factor,
                                 ewma_volume_normalization_factor, ewma_alpha,
                                 only_non_arb_ewma_volume, nn_hyper_params, vector_hyper_params))

    N_CPU = 96
    pool = ProcessingPool(nodes=N_CPU)
    job_output_list = pool.map(run_experiment, job_list)

    results = {
        'gbm_params_list': gbm_params_list,
        'non_arb_tanh_amplitude_list': non_arb_tanh_amplitude_list,
        # 'mean_non_arb_lambda_list': mean_non_arb_lambda_list,
        # 'risk_averse_a_list': risk_averse_a_list,
        # 'rebalance_cost_coeff_list': rebalance_cost_coeff_list,
        'tau_list': tau_list,
        'results': {},
    }

    for job_key, job_output in zip(job_key_list, job_output_list):
        results['results'][job_key] = job_output

    # save results
    np.save(os.path.join(dir_path, 'results_merge.npy'), results)
