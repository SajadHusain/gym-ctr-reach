"""RL-priority auxiliary gradients and finite-step checks for deterministic MLPs.

Projection is an asymmetric adaptation of PCGrad (Yu et al., NeurIPS 2020).
The checked objective is the current minibatch's fixed-critic surrogate, not
return or a physical Lyapunov function. No simulator calls occur here.
"""
from copy import deepcopy
import math

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


STEP_METRICS = (
    "gradient_dot", "gradient_cosine", "gradient_conflict", "rl_gradient_norm",
    "jacobian_parameter_gradient_norm", "weighted_aux_norm_ratio_before",
    "weighted_aux_norm_ratio_after", "projected_gradient_dot",
    "actor_step_first_order", "rl_surrogate_before", "rl_surrogate_after",
    "rl_surrogate_change", "actor_step_scale", "actor_step_rl_only",
    "actor_step_skipped", "actor_step_rejected_trials", "actor_step_checked",
)


def flatten_gradients(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    vector = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                        for p, g in zip(parameters, gradients)]).detach()
    if not bool(torch.isfinite(vector).all()):
        raise FloatingPointError("Nonfinite actor gradient")
    return vector


def combine_gradients(rl, auxiliary, weight, max_aux_ratio=1., project=True):
    """Protect only RL's direction; bound weighted auxiliary parameter norm."""
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(max_aux_ratio) or max_aux_ratio <= 0:
        raise ValueError("Weight must be nonnegative and auxiliary ratio positive, both finite")
    r, j = rl.double(), auxiliary.double()
    rr, jj, dot = r.dot(r), j.dot(j), r.dot(j)
    rn, jn = rr.sqrt(), jj.sqrt()
    adjusted = j.clone()
    if project:
        if float(rr) == 0.:
            adjusted.zero_()  # No RL descent direction to prioritize.
        elif float(dot) < 0.:
            adjusted -= dot/rr*r
        norm = weight*adjusted.norm()
        if float(norm) > float(max_aux_ratio*rn):
            adjusted *= max_aux_ratio*rn/norm
    ratio_before = float(weight*jn/rn) if float(rn) else 0.
    ratio_after = float(weight*adjusted.norm()/rn) if float(rn) else 0.
    metrics = dict(gradient_dot=float(dot),
        gradient_cosine=float(dot/(rn*jn)) if float(rn*jn) else 0.,
        gradient_conflict=float(dot < 0), rl_gradient_norm=float(rn),
        jacobian_parameter_gradient_norm=float(jn),
        weighted_aux_norm_ratio_before=ratio_before,
        weighted_aux_norm_ratio_after=ratio_after,
        projected_gradient_dot=float(r.dot(adjusted)))
    return (r+weight*adjusted).to(rl.dtype), metrics


def _set_gradients(parameters, flat):
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset+p.numel()].reshape_as(p).to(p).clone()
        offset += p.numel()


def actor_step(parameters, optimizer, rl_gradient, direction, loss_closure, *,
               checked=True, max_backtracks=6):
    """Try Adam, halve its displacement, then try RL-only Adam or skip.

Trials use the SAME critic and minibatch. Acceptance requires both a negative
actual g_R dot parameter displacement and a finite, nonincreasing surrogate.
Rejected candidates restore parameters AND optimizer moments/step counters.
Accepted fractional steps keep the candidate's moments but shorten its parameter
displacement. This is an explicit line-search variant of Adam, not plain Adam.
Only deterministic, stateless forward computations are supported by the caller.
"""
    if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    parameters = tuple(parameters)
    original = parameters_to_vector(parameters).detach().clone()
    with torch.no_grad():
        before = float(loss_closure())
    if not math.isfinite(before):
        raise FloatingPointError("Nonfinite RL surrogate before actor step")
    metrics = dict(actor_step_first_order=0., rl_surrogate_before=before,
        rl_surrogate_after=before, rl_surrogate_change=0., actor_step_scale=0.,
        actor_step_rl_only=0., actor_step_skipped=1., actor_step_rejected_trials=0.,
        actor_step_checked=float(checked))
    if checked and not bool(torch.count_nonzero(rl_gradient)):
        optimizer.zero_grad(set_to_none=True)
        return metrics
    state = deepcopy(optimizer.state_dict()) if checked else None

    def restore():
        with torch.no_grad():
            vector_to_parameters(original.clone(), parameters)
        optimizer.load_state_dict(deepcopy(state))
        optimizer.zero_grad(set_to_none=True)

    try:
        for fallback, gradient in enumerate((direction, rl_gradient) if checked else (direction,)):
            if fallback:
                restore()
            _set_gradients(parameters, gradient)
            optimizer.step()
            candidate = parameters_to_vector(parameters).detach().clone()
            displacement = candidate-original
            for halving in range(max_backtracks+1 if checked else 1):
                scale = 2.**(-halving)
                with torch.no_grad():
                    vector_to_parameters((original+scale*displacement).clone(), parameters)
                    actual = parameters_to_vector(parameters).detach()-original
                    first_order = float(rl_gradient.double().dot(actual.double()))
                    after = float(loss_closure())
                finite = math.isfinite(after) and math.isfinite(first_order)
                if finite and (not checked or (first_order < 0. and after <= before)):
                    metrics.update(actor_step_first_order=first_order, rl_surrogate_after=after,
                        rl_surrogate_change=after-before, actor_step_scale=scale,
                        actor_step_rl_only=float(fallback), actor_step_skipped=0.)
                    return metrics
                metrics["actor_step_rejected_trials"] += 1.
            if not checked:
                raise FloatingPointError("Nonfinite unchecked actor step")
        restore()
        return metrics
    except Exception:
        if checked:
            restore()
        raise
