"""Cached hydraulic topology labels, independent of boundary/source values."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def hydraulic_component_labels_3d(*, faces, active, topology_cache=None):
    act = np.asarray(active) != 0
    positive = tuple(np.asarray(faces[index]) > 0 for index in (0,2,4))
    cached = None if topology_cache is None else topology_cache.get("components")
    reused = cached is not None and np.array_equal(cached["active"],act) and all(
        np.array_equal(old,new) for old,new in zip(cached["positive"],positive))
    if reused:
        return cached["labels"],cached["count"],cached["sizes"],True
    indices = np.arange(act.size).reshape(act.shape)
    rows,columns = [],[]
    for axis,edges in zip((2,1,0),positive):
        here,there = [slice(None)]*3,[slice(None)]*3
        here[axis],there[axis] = slice(None,-1),slice(1,None)
        here,there = tuple(here),tuple(there)
        connected = edges[here]&act[here]&act[there]
        rows.append(indices[here][connected])
        columns.append(indices[there][connected])
    row,column = np.concatenate(rows),np.concatenate(columns)
    graph = coo_matrix((np.ones(row.size),(row,column)),shape=(act.size,act.size)).tocsr()
    count,labels = connected_components(csgraph=graph,directed=False)
    labels = labels.reshape(act.shape)
    sizes = np.bincount(labels[act],minlength=count)
    if topology_cache is not None:
        topology_cache["components"] = dict(active=act.copy(),positive=positive,
            labels=labels,count=count,sizes=sizes)
    return labels,count,sizes,False
