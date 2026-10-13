"""TRACER 2.0 bare chassis envelope; base_link assumed at geometric centre.

AgileX TRACER 2.0 User Manual V1.0.0 (2025-03), section 1.2:
702 x 610 x 169 mm. Accessories are not included in this envelope.
"""
HALF_LENGTH = .351
HALF_WIDTH = .305
PLANNING_PADDING = .05
SELF_FILTER_PADDING = .02
GOAL_XY_TOLERANCE = .08


def footprint(padding=0.):
    x=HALF_LENGTH+padding;y=HALF_WIDTH+padding
    return [[-x,-y],[-x,y],[x,y],[x,-y]]
