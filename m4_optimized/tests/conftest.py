import asyncio
import inspect
import pytest

def pytest_pyfunc_call(pyfuncitem):
    """Run async test functions synchronously using asyncio.run."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        testargs = {
            arg: pyfuncitem.funcargs[arg]
            for arg in pyfuncitem._fixtureinfo.argnames
            if arg in pyfuncitem.funcargs
        }
        asyncio.run(pyfuncitem.obj(**testargs))
        return True

