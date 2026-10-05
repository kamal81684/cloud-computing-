"""
workflow_parser.py
------------------
Reads a scientific workflow in Pegasus DAX (XML) format and turns it into a DAG.

From the XML we take:
  <job>              -> a task and its runtime (seconds)
  <child>/<parent>   -> a dependency edge parent -> child
  <uses> file sizes  -> data on an edge = files the parent writes and the child reads
"""

import os
import xml.etree.ElementTree as ET
from collections import deque


class Workflow:
    """A DAG. Tasks are numbered 0..n-1."""

    def __init__(self, name):
        self.name = name
        self.runtime = []    # runtime[i]   = reference runtime of task i (seconds)
        self.parents = []    # parents[i]   = list of parent tasks of i
        self.children = []   # children[i]  = list of child tasks of i
        self.data = {}       # data[(p, c)] = bytes sent from task p to task c
        self.topo = []       # tasks in topological order

    @property
    def n(self):
        return len(self.runtime)


def _tag(elem):
    return elem.tag.split("}")[-1]          # '{namespace}job' -> 'job'


def topological_order(wf):
    indeg = [len(p) for p in wf.parents]
    queue = deque(i for i in range(wf.n) if indeg[i] == 0)
    order = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for c in wf.children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    return order


def parse_dax(path):
    """Parse a DAX XML file and return a Workflow."""
    root = ET.parse(path).getroot()
    wf = Workflow(os.path.splitext(os.path.basename(path))[0])
    index, inputs, outputs = {}, [], []

    # 1) tasks
    for job in root:
        if _tag(job) != "job":
            continue
        index[job.get("id")] = wf.n
        wf.runtime.append(float(job.get("runtime", 0)))
        wf.parents.append([])
        wf.children.append([])
        ins, outs = {}, {}
        for use in job:
            if _tag(use) == "uses":
                name = use.get("file") or use.get("name")
                size = float(use.get("size", 0))
                (outs if use.get("link") == "output" else ins)[name] = size
        inputs.append(ins)
        outputs.append(outs)

    # 2) dependencies + data on each edge
    for child in root:
        if _tag(child) != "child":
            continue
        c = index[child.get("ref")]
        for parent in child:
            p = index[parent.get("ref")]
            if p in wf.parents[c]:
                continue
            wf.parents[c].append(p)
            wf.children[p].append(c)
            shared_files = set(outputs[p]) & set(inputs[c])
            wf.data[(p, c)] = sum(outputs[p][f] for f in shared_files)

    wf.topo = topological_order(wf)
    return wf
