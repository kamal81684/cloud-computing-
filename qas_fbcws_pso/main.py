"""
main.py
-------
Sets up the cloud, runs the chosen scheduling algorithms on each workflow
and saves all metrics to results.csv.

Run:
    python main.py                      # uses sample_workflow.xml
    python main.py a.xml b.xml ...      # any Pegasus DAX workflow files
"""

import csv
import random
import sys
import time

import fbcws
import pso
from metrics import evaluate, heft_makespan
from workflow_parser import parse_dax

# ---------------------------------- settings ----------------------------------
ALGORITHMS = {
    "FBCWS":     lambda inst: fbcws.schedule(inst),                   # original paper
    "FBCWS-CA":  lambda inst: fbcws.schedule(inst, comm_aware=True),  # + CRI communication-aware
    "FBCWS-PSO": lambda inst: pso.schedule(inst),                     # + PSO (proposed)
}
VM_COUNTS = [5, 10, 20]          # number of VMs to test
BUDGET_FACTORS = [0.2, 0.5]      # budget = Costmin + k * (Costmax - Costmin)
BANDWIDTH = 20e6 / 8             # 20 Mbps in bytes/s (paper, Sec. 5.1)
SEED = 1
OUTPUT_CSV = "results.csv"
# ------------------------------------------------------------------------------


class Instance:
    """Workflow + cloud = everything a scheduler needs.
    et[i][j]    execution time of task i on VM j
    cost[i][j]  price of VM j x et[i][j]                        (paper Eq. 1)
    comm[(p,c)] transfer time of edge p->c if on different VMs"""

    def __init__(self, wf, n_vms, seed=SEED):
        rng = random.Random(seed)
        self.wf, self.n, self.v = wf, wf.n, n_vms

        # VMs: faster ones cost more per unit of work -> time vs cost trade-off
        speeds = sorted(rng.uniform(1.0, 4.0) for _ in range(n_vms))
        self.price = [0.10 * s ** 1.5 * rng.uniform(0.9, 1.1) / 3600 for s in speeds]  # $/s

        # +-15% task-VM variation: a VM is not equally good for every task
        self.et = [[wf.runtime[i] / s * rng.uniform(0.85, 1.15) for s in speeds]
                   for i in range(wf.n)]
        self.cost = [[self.et[i][j] * self.price[j] for j in range(n_vms)] for i in range(wf.n)]
        self.comm = {edge: size / BANDWIDTH for edge, size in wf.data.items()}

        self.cost_min = sum(min(row) for row in self.cost)    # Eq. 3
        self.cost_max = sum(max(row) for row in self.cost)    # Eq. 4
        self.budget = None

    def set_budget(self, k):
        self.budget = self.cost_min + k * (self.cost_max - self.cost_min)


def main(files):
    rows = []
    for path in files:
        wf = parse_dax(path)
        for n_vms in VM_COUNTS:
            inst = Instance(wf, n_vms)
            heft_mk = heft_makespan(inst, fbcws.priority_order(inst))
            for k in BUDGET_FACTORS:
                inst.set_budget(k)
                for name, algorithm in ALGORITHMS.items():
                    start = time.time()
                    order, vm_of = algorithm(inst)
                    row = {"workflow": wf.name, "tasks": wf.n, "vms": n_vms,
                           "budget_factor": k, "budget": round(inst.budget, 5),
                           "algorithm": name}
                    row.update(evaluate(inst, order, vm_of, heft_mk))
                    row["run_time_s"] = round(time.time() - start, 3)
                    rows.append(row)
                    print(f"{wf.name:15s} vms={n_vms:<3} k={k:<4} {name:10s} "
                          f"makespan={row['makespan']:<10} cost={row['cost']:<9} "
                          f"comm={row['comm_time']:<9} budget_met={row['budget_met']}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["sample_workflow.xml"])
