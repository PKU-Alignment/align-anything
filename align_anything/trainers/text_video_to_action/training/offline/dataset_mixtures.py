import sys


def get_mixture_by_name(name):
    return getattr(sys.modules[__name__], name, [name])


CHORES = [
    # "ObjectNavType",
    "PickupType",
    # "FetchType",
    # "RoomVisit",  # "SimpleExploreHouse",  #
]

CHORESNAV = [
    "ObjectNavType",
    "ObjectNavRoom",
    "ObjectNavRelAttribute",
    "ObjectNavAffordance",
    "ObjectNavLocalRef",
    "ObjectNavDescription",
    "RoomNav",
]
