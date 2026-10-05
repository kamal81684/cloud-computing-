"""
metrics.py
----------
Given a schedule = (task order, VM of each task), compute:

  makespan  = finish time of the last task                          (paper Eq. 6)
  cost      = sum of price x execution time                         (Eq. 2)
  comm_time = total data-transfer time between tasks on different VMs
  NM        = makespan / makespan of HEFT                           (Eq. 23)
  NC        = cost / budget   (NC <= 1 means the budget is met)     (Eq. 24)
"""


def simulate(inst, order, vm_of):
    """Run tasks in `order` on their VMs. A task starts when its VM is free
    AND the data from all its parents has arrived (Eqs. 20-22)."""
    vm_free = [0.0] * inst.v
    finish = [0.0] * inst.n
    for i in order:
        j = vm_of[i]
        data_ready = 0.0
        for p in inst.wf.parents[i]:
            transfer = 0.0 if vm_of[p] == j else inst.comm[(p, i)]   # same VM -> no transfer
            data_ready = max(data_ready, finish[p] + transfer)
        start = max(vm_free[j], data_ready)
        finish[i] = start + inst.et[i][j]
        vm_free[j] = finish[i]
    return finish


def makespan(inst, order, vm_of):
    return max(simulate(inst, order, vm_of))


def total_cost(inst, vm_of):
    return sum(inst.cost[i][vm_of[i]] for i in range(inst.n))


def communication_cost(inst, vm_of):
    """Total transfer time (s) and data moved (MB) between tasks on different VMs."""
    time, data = 0.0, 0.0
    for (p, c), t in inst.comm.items():
        if vm_of[p] != vm_of[c]:
            time += t
            data += inst.wf.data[(p, c)]
    return time, data / 1e6


def heft_makespan(inst, order):
    """Makespan of HEFT: every task goes to the VM where it finishes earliest
    (budget ignored). Only used as the reference value for NM."""
    vm_of = [None] * inst.n
    finish = [0.0] * inst.n
    vm_free = [0.0] * inst.v
    for i in order:
        best_finish, best_vm = None, None
        for j in range(inst.v):
            data_ready = max((finish[p] + (0.0 if vm_of[p] == j else inst.comm[(p, i)])
                              for p in inst.wf.parents[i]), default=0.0)
            f = max(vm_free[j], data_ready) + inst.et[i][j]
            if best_finish is None or f < best_finish:
                best_finish, best_vm = f, j
        finish[i], vm_of[i] = best_finish, best_vm
        vm_free[best_vm] = best_finish
    return max(finish)


def evaluate(inst, order, vm_of, heft_mk):
    """All metrics of one schedule, as one CSV row."""
    mk = makespan(inst, order, vm_of)
    cost = total_cost(inst, vm_of)
    comm_time, comm_mb = communication_cost(inst, vm_of)
    return {
        "makespan": round(mk, 3),
        "cost": round(cost, 5),
        "comm_time": round(comm_time, 3),
        "data_moved_MB": round(comm_mb, 2),
        "NM": round(mk / heft_mk, 4),
        "NC": round(cost / inst.budget, 4),
        "budget_met": cost <= inst.budget + 1e-9,
    }
