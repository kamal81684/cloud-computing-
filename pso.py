"""
pso.py
------
FBCWS-PSO (proposed): instead of choosing a VM for each LTCTL task one at a time,
Particle Swarm Optimization searches VM choices for ALL LTCTL tasks together.

Unchanged from FBCWS: categorization, B-level order, budget, MTCTL rule.

Particle : one number per LTCTL task in [0, v)  ->  int(x) = chosen VM
Decode   : run the FBCWS pass; LTCTL tasks take the particle's VM
           (if that VM is over the task's budget, use the cheapest VM instead,
            so every particle gives a schedule that meets the budget)
Fitness  : (makespan, comm_time, cost), compared in that order.
           The makespan comes from simulating the whole workflow, so it
           already includes data-transfer delays caused by the VM choices.
Seeding  : two particles start at the FBCWS and FBCWS-CA solutions,
           so PSO is never worse than either of them.
"""

import numpy as np

import fbcws
from metrics import makespan, total_cost, communication_cost

SWARM_SIZE = 30
ITERATIONS = 100
PATIENCE = 30              # stop early if no improvement for this many iterations
W_START, W_END = 0.9, 0.4  # inertia weight, decreases linearly
C1, C2 = 1.5, 1.5          # personal-best and global-best pull


def decode(inst, order, mtctl, ltctl_index, particle_vms):
    """Particle -> full VM mapping that respects the budget."""
    budget = fbcws.Budget(inst)
    vm_of = [None] * inst.n
    for i in order:
        vms = budget.affordable_vms(i)
        if i in mtctl:
            j = fbcws.fastest(inst, i, vms)
        else:
            j = particle_vms[ltctl_index[i]]
            if j not in vms:                           # over budget -> repair
                j = fbcws.cheapest(inst, i, vms)
        budget.pay(i, j)
        vm_of[i] = j
    return vm_of


def schedule(inst, seed=0):
    order = fbcws.priority_order(inst)
    mtctl = fbcws.categorize(inst)
    ltctl = [i for i in order if i not in mtctl]
    ltctl_index = {t: k for k, t in enumerate(ltctl)}
    dim, v = len(ltctl), inst.v
    if dim == 0 or v == 1:
        return fbcws.schedule(inst)

    cache = {}

    def evaluate(position):
        """Fitness of a particle position (cached, many particles repeat)."""
        key = tuple(np.minimum(position.astype(int), v - 1))
        if key not in cache:
            vm_of = decode(inst, order, mtctl, ltctl_index, key)
            fit = (round(makespan(inst, order, vm_of), 6),
                   round(communication_cost(inst, vm_of)[0], 6),
                   total_cost(inst, vm_of))
            cache[key] = (fit, vm_of)
        return cache[key]

    # initial swarm: random particles + 2 seeded with the greedy solutions
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, v, (SWARM_SIZE, dim))
    for k, comm_aware in enumerate([False, True]):
        _, vm_of = fbcws.schedule(inst, comm_aware)
        X[k] = [vm_of[t] + 0.5 for t in ltctl]
    v_max = max(1.0, 0.25 * v)
    V = rng.uniform(-v_max, v_max, (SWARM_SIZE, dim))

    pbest = X.copy()
    pbest_fit = [evaluate(x)[0] for x in X]
    g = min(range(SWARM_SIZE), key=lambda k: pbest_fit[k])
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]

    stall = 0
    for it in range(ITERATIONS):
        w = W_START - (W_START - W_END) * it / ITERATIONS
        r1, r2 = rng.random((SWARM_SIZE, dim)), rng.random((SWARM_SIZE, dim))
        V = np.clip(w * V + C1 * r1 * (pbest - X) + C2 * r2 * (gbest - X), -v_max, v_max)
        X = np.clip(X + V, 0, v - 1e-9)

        improved = False
        for k in range(SWARM_SIZE):
            fit, _ = evaluate(X[k])
            if fit < pbest_fit[k]:
                pbest[k], pbest_fit[k] = X[k].copy(), fit
                if fit < gbest_fit:
                    gbest, gbest_fit = X[k].copy(), fit
                    improved = True
        stall = 0 if improved else stall + 1
        if stall >= PATIENCE:
            break

    return order, evaluate(gbest)[1]
