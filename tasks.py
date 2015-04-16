"""Using `invoke` to start tests and and other commandline tools.
"""

from invoke import run, task


@task
def test():
    """Run standard tests.


    Usage (commandline): inv test
    """
    run('py.test --assert=reinterp', pty=True)
