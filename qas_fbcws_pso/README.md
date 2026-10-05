# QoS-Aware Workflow Scheduling: FBCWS + Communication-Aware PSO

Cloud Computing course project.

**Base paper:** N. Rizvi, D. Ramesh, *Fair budget constrained workflow scheduling approach for heterogeneous clouds*, Cluster Computing 23, 3185–3201 (2020).

## The paper (FBCWS)

FBCWS schedules a workflow on cloud VMs to minimize makespan while staying within a budget. It splits tasks into two lists:

- **MTCTL** (time-consuming tasks) go to the fastest VM within budget.
- **LTCTL** (normal tasks) go to the VM that minimizes `β·Time + (1−β)·Cost`.

## The gap

1. LTCTL VMs are chosen **greedily, one task at a time**, so a better combination of choices can be missed.
2. VM selection uses **only time and cost**. Data transfer between tasks is ignored, and the paper itself reports that FBCWS is weak on data-intensive workflows (Montage, CyberShake).

## What we add

**1. Communication-aware VM selection (FBCWS-CA).** For each task:

```
Comm_i = Σ Data(p,i) / Bandwidth        over all parents p
CRI_i  = Comm_i / (ACT_i + Comm_i)      Communication Ratio Index

Score  = w_t·NT + w_c·NC + w_comm·NComm
```

`NComm` is the transfer time the task waits for on that VM. Data from a parent already on the same VM needs no transfer. The weights depend on the task's CRI:

| Task type | CRI | w_t | w_c | w_comm |
|---|---|---|---|---|
| Computation-intensive | < 0.5 | 0.60 | 0.25 | 0.15 |
| Communication-intensive | ≥ 0.5 | 0.35 | 0.20 | 0.45 |

**2. PSO for LTCTL tasks (FBCWS-PSO).** PSO searches VM choices for **all LTCTL tasks together**. Each candidate is scored by simulating the whole workflow. The score compares makespan first, then communication time, then cost. Over-budget choices are replaced with the cheapest VM, so the budget is always met. PSO starts from the FBCWS and FBCWS-CA solutions, so it is never worse than them.

## Files

| File | What it does |
|---|---|
| `main.py` | Sets up the VMs, runs the algorithms, saves `results.csv` |
| `workflow_parser.py` | Reads a workflow XML (Pegasus DAX) into tasks, edges and data sizes |
| `metrics.py` | Computes makespan, cost, communication cost, NM and NC |
| `fbcws.py` | FBCWS from the paper, plus the communication-aware option |
| `pso.py` | PSO version (proposed) |
| `sample_workflow.xml` | A 25-task Montage workflow |

## How to run

```
pip install numpy
python main.py                              # runs on sample_workflow.xml
python main.py Montage_100.xml Sipht_60.xml # runs on any DAX workflows
```

More workflows (Montage, CyberShake, Epigenomics, Inspiral, Sipht) are available in the `config/dax` folder of the WorkflowSim repository: https://github.com/WorkflowSim/WorkflowSim-1.0

## Output: `results.csv`

One row per workflow × VM count × budget × algorithm.

| Column | Meaning |
|---|---|
| makespan | Finish time of the last task (s) |
| cost | Total execution cost ($) |
| comm_time | Total data-transfer time between different VMs (s) |
| data_moved_MB | Data sent between different VMs |
| NM | makespan / HEFT makespan (paper Eq. 23); lower is better |
| NC | cost / budget (paper Eq. 24); ≤ 1 means the budget is met |
| budget_met | True if cost ≤ budget |

## Settings and assumptions

Settings are at the top of `main.py`, `fbcws.py` and `pso.py`:

- **VMs:** 5, 10 and 20, with random speeds (seeded). Faster VMs cost more per unit of work, and each task runs ±15% differently on each VM.
- **Budget:** `Costmin + k·(Costmax − Costmin)` with k = 0.2 and 0.5.
- **Bandwidth:** 20 Mbps, the same for all VMs (as in the paper). Data transfer has no monetary cost.
- **PSO:** 30 particles, up to 100 iterations, inertia 0.9 → 0.4, c1 = c2 = 1.5.
- **Check:** our FBCWS reproduces the paper's worked example exactly (Table 10: makespan 80, cost 471; Table 11: makespan 90, cost 455).
