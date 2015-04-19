.. -*- rest -*-

Mochi Testing Guidelines
========================

:version: 0.1
:authors: Mike Müller

.. sidebar:: Work in Progress

    This guide is work in progress. There will likely b changes to
    procedures described below. Furthermore, there will additional tests
    and test procedure for other test types. This would include testing
    on the s-expression level or testing during the parsing and lexing
    phase. So far this guide only covers higher level test, i.e. functions
    written in Mochi against expected results.

.. contents::

Introduction
------------

Since Mochi is based on Python, compiles to Python bytecode, can import and
use Python libraries, and can be imported by Python, we use the mature
and feature-rich `py.test <http://pytest.org>`__ test runner.
We also going to use `tox <https://tox.readthedocs.org>`__
that works very well with `py.test`.


High-Level Mochi Tests
----------------------

Invoking Tests
++++++++++++++

Typing `inv test` the directory main directory, i.e. the directory that
contains the `tasks.py` will start the tests. Currently, this runs
`py.test --assert=reinterp`.
The command line option is need to make bytecode-only Python modules work.

Mochi-only Tests
++++++++++++++++

You can run test by creating a Mochi file following these steps:

1. Create a file `mochi_test_*.mochi` inside the directory `tests`.
2. Place function `<func>` you would to test insides this file.
3. Put another function with the name `result_<func>` into the same file.
   This function needs to returns the expected result.
4. Run `inv test` from the main directory.
5. Change the return value of `result_<func>` to a wrong value to see how a
   failed test looks like.

Example
'''''''

This is a small example:

.. code-block:: python

    # file mochi_test_add.mochi in dir tests

    def add():
        return 1 + 1


    def result_add():
        return 2   # Change to 3 to see how a test fails.

Your test function cannot have parameters. However, you can call
other functions inside your test function. For example:

.. code-block:: python

    def test_concat():
        concat([1, 2 ,3], [4, 5, 6])

    def result_test_concat():
        pvector([1, 2, 3, 4, 5, 6])


You may put as many functions `result_<func>` in one Mochi file.
Each needs to have a corresponding function `<func>` inside the same file.
In addition, you may define other functions in such a file. They will be
ignore by the test runner as long as their names do not follow the pattern
`result_<func>`. This allows you to write helper functions. One use case is
to call other functions with arguments as the tested

Mixed Mochi-Python Tests
++++++++++++++++++++++++

Mochi-only tests are somewhat restricted and you cannot use all `py.test`
features. Therefore, you can writing mixed Mochi-Python tests.
This always involves a Mochi and Python file, but gives you all the power of
`py.test`.

1. Create a Mochi file with the functions you would like to test.
2. Create a Python file `test_<name>.py`.
3. In this file import `mochi.utils.pycloader.get_module`.
4. Import the Mochi file as Python module with
   `mod_<name> = get_module('<name>', file_path=__file__)`
5. Use the Mochi functions in your test, referencing them as
   `mod_<name>.<mochi_func>`.
6. Run `inv test` from the main directory.
7. Change your test so that `assert` fails to see how a failed
   test looks like.

Let's have a look at an example. This is our Mochi file:

.. code-block:: python

    # file add.mochi in dir tests

    def add(a, b):
        return a + b

    def add10(a):
        return a + 10

The Python test file looks like this:

.. code-block:: python

    # file test_add.py in dir tests


    from mochi.utils.pycloader import get_module


    mod_add = get_module('add', file_path=__file__)


    def test_add():
        assert mod_add.add(2, 2) == 4


    def test_add10():
        assert mod_add.add10(2) == 12 # Change to 10 to see how a test fails.








