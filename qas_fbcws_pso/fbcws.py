"""
fbcws.py
--------
FBCWS (Rizvi & Ramesh, Cluster Computing 2020) + our communication-aware option.

Step 1  Categorize : task with ACT >= mean ACT of its level -> MTCTL, else LTCTL   (Eqs. 8-10)
Step 2  Priority   : B-level, highest first                                        (Eq. 11)
Step 3  Budget     : each task may spend at most BC = RB - RCB                     (Eqs. 12-15)
Step 4  VM choice  :
        MTCTL -> fastest VM within budget
        LTCTL -> comm_aware=False (paper):  min  beta*NT + (1-beta)*NC             (Eq. 16)
                 comm_aware=True  (ours) :  min  w_t*NT + w_c*NC + w_comm*NComm
                                            weights picked by the task's CRI
"""

BETA = 0.8                              # paper's cost-time factor
CRI_THRESHOLD = 0.5                     # CRI >= 0.5 -> communication-intensive task
WEIGHTS_COMPUTE = (0.60, 0.25, 0.15)    # (w_t, w_c, w_comm) computation-intensive
WEIGHTS_COMM = (0.35, 0.20, 0.45)       # (w_t, w_c, w_comm) communication-intensive
EPS = 1e-9


# ----------------------------- Step 1: categorize -----------------------------

def act(inst):
    """Average computation time of each task over all VMs (Eq. 8)."""
    return [sum(row) / inst.v for row in inst.et]


def categorize(inst):
    """Return the set of MTCTL tasks (everything else is LTCTL)."""
    a = act(inst)
    level = [1] * inst.n                                   # Eq. 9
    for i in inst.wf.topo:
        for p in inst.wf.parents[i]:
            level[i] = max(level[i], level[p] + 1)
    per_level = {}
    for i in range(inst.n):
        per_level.setdefault(level[i], []).append(a[i])
    mact = {l: sum(v) / len(v) for l, v in per_level.items()}   # Eq. 10
    return {i for i in range(inst.n) if a[i] >= mact[level[i]] - EPS}


# ----------------------------- Step 2: priority -------------------------------

def b_level(inst):
    """Longest path (computation + communication) from a task to the exit (Eq. 11)."""
    a = act(inst)
    bl = [0.0] * inst.n
    for i in reversed(inst.wf.topo):
        bl[i] = a[i] + max((inst.comm[(i, c)] + bl[c] for c in inst.wf.children[i]), default=0.0)
    return bl


def priority_order(inst):
    bl = b_level(inst)
    pos = {t: k for k, t in enumerate(inst.wf.topo)}
    return sorted(range(inst.n), key=lambda i: (-round(bl[i], 6), pos[i]))


# ----------------------------- Step 3: budget ---------------------------------

class Budget:
    """RB  = budget not spent yet
    RCB = cheapest possible cost of the tasks not scheduled yet
    A task may spend BC = RB - RCB, so every later task can still afford
    its cheapest VM -> the final schedule always meets the budget."""

    def __init__(self, inst):
        self.inst = inst
        self.rb = inst.budget
        self.rcb = inst.cost_min

    def affordable_vms(self, i):
        """Call once per task, in priority order."""
        self.rcb -= min(self.inst.cost[i])                      # Eq. 12
        bc = self.rb - self.rcb                                 # Eq. 14
        ok = [j for j in range(self.inst.v) if self.inst.cost[i][j] <= bc + EPS]
        return ok or [cheapest(self.inst, i, range(self.inst.v))]

    def pay(self, i, j):
        self.rb -= self.inst.cost[i][j]                         # Eq. 13


# ----------------------------- Step 4: VM choice ------------------------------

def fastest(inst, i, vms):
    return min(vms, key=lambda j: (inst.et[i][j], inst.cost[i][j]))


def cheapest(inst, i, vms):
    return min(vms, key=lambda j: (inst.cost[i][j], inst.et[i][j]))


def select_original(inst, i, vms):
    """Paper: take a VM that is both fastest and cheapest, else minimize Eq. 16."""
    f, c = fastest(inst, i, vms), cheapest(inst, i, vms)
    if f == c:
        return f
    et_max, cost_max = max(inst.et[i]) or 1.0, max(inst.cost[i]) or 1.0
    return min(vms, key=lambda j: BETA * inst.et[i][j] / et_max
                                  + (1 - BETA) * inst.cost[i][j] / cost_max)


def cri(inst):
    """Comm_i = sum of Data(p,i)/Bandwidth over parents (all-remote case)
       CRI_i  = Comm_i / (ACT_i + Comm_i)        Communication Ratio Index"""
    a = act(inst)
    comm = [sum(inst.comm[(p, i)] for p in inst.wf.parents[i]) for i in range(inst.n)]
    ratio = [comm[i] / (a[i] + comm[i]) if a[i] + comm[i] > 0 else 0.0 for i in range(inst.n)]
    return ratio, comm


def select_comm_aware(inst, i, vms, vm_of, cri_i, comm_i):
    """Ours: Score = w_t*NT + w_c*NC + w_comm*NComm, weights chosen by CRI.
    NComm = transfer time if task i runs on VM j / Comm_i
            (data from a parent already on VM j needs no transfer)."""
    w_t, w_c, w_comm = WEIGHTS_COMM if cri_i >= CRI_THRESHOLD else WEIGHTS_COMPUTE
    et_max, cost_max = max(inst.et[i]) or 1.0, max(inst.cost[i]) or 1.0

    def score(j):
        transfer = sum(inst.comm[(p, i)] for p in inst.wf.parents[i] if vm_of[p] != j)
        n_comm = transfer / comm_i if comm_i > 0 else 0.0
        return w_t * inst.et[i][j] / et_max + w_c * inst.cost[i][j] / cost_max + w_comm * n_comm

    return min(vms, key=score)


# ----------------------------- the algorithm ----------------------------------

def schedule(inst, comm_aware=False):
    """Returns (task order, VM of each task)."""
    order = priority_order(inst)
    mtctl = categorize(inst)
    cri_all, comm_all = cri(inst)
    budget = Budget(inst)
    vm_of = [None] * inst.n

    for i in order:
        vms = budget.affordable_vms(i)
        if i in mtctl:
            j = fastest(inst, i, vms)
        elif comm_aware:
            j = select_comm_aware(inst, i, vms, vm_of, cri_all[i], comm_all[i])
        else:
            j = select_original(inst, i, vms)
        budget.pay(i, j)
        vm_of[i] = j

    return order, vm_of
