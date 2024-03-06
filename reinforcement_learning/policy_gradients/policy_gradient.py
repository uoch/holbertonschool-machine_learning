import numpy as np


def softmax(logits):
    """softmax function with normalization"""
    # Normalize logits for numerical stability
    normalized_logits = logits - np.max(logits)
    exp_logits = np.exp(normalized_logits)
    return exp_logits / np.sum(exp_logits)


def softmax_grad(s):
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m


def policy(state, weight):
    """simple policy function"""
    # flatten the state with the weight
    logits = np.dot(state, weight)
    action_probs = softmax(logits)
    if np.isnan(action_probs).any():
        return np.ones_like(action_probs) / len(action_probs)
    return action_probs


def policy_gradient(state, weight):
    logits = np.dot(state, weight)
    action_probs = softmax(logits)
    action = np.random.choice(len(action_probs), p=action_probs)
    if np.isnan(action_probs).any():
        return np.ones_like(action_probs) / len(action_probs)
    grad_log_probs = np.outer(
        state, (action_probs - (action == np.arange(len(action_probs)))))

    return action, grad_log_probs
