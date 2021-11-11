import asyncio
import inspect
import sys


def is_double_callable(application):
    """
    Tests to see if an application is a legacy-style (double-callable) application.
    """
    # Look for a hint on the object first
    if getattr(application, "_asgi_single_callable", False):
        return False
    if getattr(application, "_asgi_double_callable", False):
        return True
    # Uninstanted classes are double-callable
    if inspect.isclass(application):
        return True
    # Instanted classes depend on their __call__
    if hasattr(application, "__call__"):
        # We only check to see if its __call__ is a coroutine function -
        # if it's not, it still might be a coroutine function itself.
        if asyncio.iscoroutinefunction(application.__call__):
            return False
    # Non-classes we just check directly
    return not asyncio.iscoroutinefunction(application)


def double_to_single_callable(application):
    """
    Transforms a double-callable ASGI application into a single-callable one.
    """

    async def new_application(scope, receive, send):
        instance = application(scope)
        return await instance(receive, send)

    return new_application


def guarantee_single_callable(application):
    """
    Takes either a single- or double-callable application and always returns it
    in single-callable style. Use this to add backwards compatibility for ASGI
    2.0 applications to your server/test harness/etc.
    """
    if is_double_callable(application):
        application = double_to_single_callable(application)
    return application


if sys.version_info >= (3, 7):
    # these were introduced in 3.7
    get_running_loop = asyncio.get_running_loop
    run_future = asyncio.run
    create_task = asyncio.create_task
else:
    # marked as deprecated in 3.10, did not exist before 3.7
    get_running_loop = asyncio.get_event_loop
    run_future = asyncio.ensure_future
    # does nothing, this is fine for <3.7
    create_task = lambda task: task
