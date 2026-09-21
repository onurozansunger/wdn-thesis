"""Frozen composition of received trajectory and flow evidence, no fitted gate."""
import numpy as np
from wdn.received_trajectory import received_trajectory_features
from wdn.received_flow_context import flow_context_features


def received_joint_features(reference, values, mask, flow, flow_mask, timestep,
                            query_time, query_node, query_scale, rounding_m=0.):
    trajectory, names = received_trajectory_features(
        values, mask, timestep, query_time, query_node, query_scale, rounding_m)
    context, added = flow_context_features(
        reference, values, mask, flow, flow_mask, timestep, query_time,
        query_node, query_scale, rounding_m)
    return np.column_stack((trajectory, context)), names + added
