"""Simulation backend selector for GAIT."""

from habitat_test import Runner, AsyncRunners, ExtendedTimeStep, make_async_runners


def get_env_classes(simulator: str):
    simulator = (simulator or "habitat-sim").lower()
    if simulator == "unrealzoo":
        from unrealzoo_test import AestheticTourDMCWrapper, MultiSceneWrapper
    else:
        from habitat_test import AestheticTourDMCWrapper, MultiSceneWrapper
    return AestheticTourDMCWrapper, MultiSceneWrapper


__all__ = [
    "Runner",
    "AsyncRunners",
    "ExtendedTimeStep",
    "make_async_runners",
    "get_env_classes",
]
