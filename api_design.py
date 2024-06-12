import legate.core.types as ty

def legate_freeze():
    """
    User declares they are going to directly interact with the underlying legion runtime.
    As a result, all legate tasks are synchronously submitted to the legion runtime 
    to preserve ordering semantics and operations cannot be submitted.
    """
    pass


def legate_unfreeze():
    """
    User returns control of runtime to Legate
    """
    pass


def set_legion_compose_mode():
    """
    User declares they will interleave Legion operations with Legate.
    As a result, Legate will serialize operation submissions
    """

def unset_legion_compose_mode():
    """
    User declares they will no longer interleave Legion operations.
    As a result the solver is free to batch schedule operations.
    """  

def wrap_region_field(
    field_id,
    logical_region,
    permission_owning_logical_region_t,
    dtype: ty.int64
):
    """
    Create Legate state storing objects separate from the primary legate runtime 
    for garbage collection purposes.
    At Present not doing anything with permission_owning_logical_region_t, but
    the easiest thing to do would be to wrap a Store around it as a parent if 
    needed and leave it to the garbage collector to worry about.
    """
    pass

def extract_future():
    """
    return's the Store's backing Future while checking it is backed by Future
    """
    pass
