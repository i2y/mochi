Mochi
=====

Mochi is a dynamically typed programming language for functional
programming and actor-style programming.

Its interpreter is written in Python3. The interpreter translates a
program written in Mochi to Python3's AST / bytecode.

Features
--------

-  Python-like syntax
-  Tail recursion optimization (self tail recursion only), and no loop
   syntax
-  Re-assignment are not allowed in function definition.
-  Basic collection type is a persistent data structure. (using
   Pyrsistent)
-  Pattern matching / Data types, like algebraic data types
-  Pipeline operator
-  Syntax sugar of anonymous function definition
-  Actor, like Erlang's actor（using Eventlet)
-  Built-in Python3 itertools and functools, operator module functions
   and function in itertools recipes

Examples
--------

Factorial
~~~~~~~~~

.. code:: python

    def factorial(n, m):
        if n == 1:
            m
        else:
            factorial(n - 1, n * m)

FizzBuzz
~~~~~~~~

.. code:: python

    def fizzbuzz(n):
        match [n % 3, n % 5]:
            [0, 0]: "fizzbuzz"
            [0, _]: "fizz"
            [_, 0]: "buzz"
            _: n

    range(1, 31) |> map(fizzbuzz) |> pvector() |> print()
    # or
    range(1, 31) |> map(fizzbuzz) |> lazyseq() |> print()

Actor
~~~~~

.. code:: python

    def show():
        receive:
            message:
                print(message)
                show()

    actor = spawn(show)

    send('foo', actor)
    actor ! 'bar' # send('bar', actor)

    wait_all()

Flask
~~~~~

.. code:: python

    from flask import Flask

    app = Flask('demo')

    @app.route('/')
    def hello():
        'Hello World!'

    app.run()

Requirements
------------

-  CPython >= 3.2 or PyPy >= 3.2.1
-  rply >= 0.7.2
-  pyrsistent >= 0.6.2
-  pathlib >= 1.0.1
-  eventlet >= 0.15.2

Installation
------------

.. code:: sh

    $ pip install mochi

Usage
-----

REPL
~~~~

.. code:: sh

    $ mochi
    >>>

loading and running a file
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    $ cat kinako.mochi
    print('kinako')
    $ mochi kinako.mochi
    kinako
    $

byte compilation
~~~~~~~~~~~~~~~~

.. code:: sh

    $ mochi -c kinako.mochi > kinako.mochic

running a byte-compiled file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    $ mochi -e kinako.mochic
    kinako
    $

Examples for each feature
-------------------------

Persistent data structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    [1, 2, 3]
    # => pvector([1, 2, 3])

    v(1, 2, 3)
    # => pvector([1, 2, 3])

    vec = [1, 2, 3]
    vec2 = vec.set(0, 8)
    # => pvector([8, 2, 3]
    vec
    # => pvector([1, 2, 3])
    [x, y, z] = vec
    x # => 1
    y # => 2
    z # => 3

    {'x': 100, 'y': 200}
    # => pmap({'y': 200, 'x': 100})

    ma = {'x': 100, 'y': 200}
    ma.get('x') # => 100
    ma.x # => 100
    ma2 = ma.set('x', 10000)
    # => pmap({'y': 200, 'x': 10000})
    ma # => pmap({'y': 200, 'x': 100})

    m(x=100, y=200)
    # => pmap({'y': 200, 'x': 100})

    s(1, 2, 3)
    # => pset([1, 2, 3])

    b(1, 2, 3)
    # => pbag([1, 2, 3])

Function definitions
~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def hoge(x):
        hoge + str(x)

    hoge(3)
    # => hoge3

Pattern matching
~~~~~~~~~~~~~~~~

.. code:: python

    lis = [1, 2, 3]

    match lis:
        [1, 2, x]: x
        _: None
    # => 3

    match lis:
        [1, &rest]: rest
        _: None

    # => pvector (2, 3)

    foo_map = {'foo' : 'bar'}

    match foo_map:
        {'foo' : value}: value
        _: None
    # => 'bar'

    match 10:
        int(x): 'int'
        float(x): 'float'
        str(x): 'str'
        bool(x): 'bool'
        _: 'other'
    # => 'int'

    match [1, 2, 3]:
        [1, str(x), 3]: 'str'
        [1, int(x), 3]: 'int'
        _: 'other'
    # => 'int'

Records
~~~~~~~

.. code:: python

    record Mochi
    record AnkoMochi(anko) < Mochi
    record KinakoMochi(kinako) < Mochi

    anko_mochi = AnkoMochi(anko=3)

    isinstance(anko_mochi, Mochi)
    # => True
    isinstance(anko_mochi, AnkoMochi)
    # => True
    isinstance(anko_mochi, KinakoMochi)
    # => False

    match anko_mochi:
        KinakoMochi(kinako): 'kinako ' * kinako + ' mochi'
        AnkoMochi(anko): 'anko ' * anko + 'mochi'
        Mochi(_): 'mochi'
    # => 'anko anko anko mochi'


    record Person(name, age):
        def show(self):
            print(self.name + ': ' + self.age)

    foo = Person('foo', '32')
    foo.show()
    # -> foo: 32

Bindings
~~~~~~~~

.. code:: python

    x = 3000
    # => 3000

    [a, b] = [1, 2]
    a
    # => 1
    b
    # => 2

    [c, &d] = [1, 2, 3]
    c
    # => 1
    d
    # => pvector([2, 3])

Data types, like algebraic data types (sum type)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    data Point:
        Point2D(x, y)
        Point3D(x, y, z)

    # The meaning of the above is the same as the meaning of the following.
    # record Point
    # record Point2D(x, y) < Point
    # record Point3D(x, y, z) < Point

    p1 = Point2D(x=1, y=2)
    # => Point2D(x=1, y=2)

    p2 = Point2D(3, 4)
    # => Point2D(x=3, y=4)

    p1.x
    # => 1

Pattern-matching function definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    data Point:
        Point2D(x, y)
        Point3D(x, y, z)

    defm offset:
        [Point2D(x1, y1), Point2D(x2, y2)]:
            Point2D(x1 + x2, y1 + y2)
        [Point3D(x1, y1, z1), Point3D(x2, y2, z2)]:
            Point3D(x1 + x2, y1 + y2, z1 + z2)
        _: None

    offset(Point2D(1, 2), Point2D(3, 4))
    # => Point2D(x=4, y=6)
    offset(Point3D(1, 2, 3), Point3D(4, 5, 6))
    # => Point3D(x=5, y=7, z=9)

Anonymous function
~~~~~~~~~~~~~~~~~~

.. code:: python

    # Arrow expression.
    add = (x, y) -> x + y
    add(1, 2)
    # => 3

    add = -> $1 + $2
    add(1, 2)
    # => 3

    foo = (x, y) ->
        if x == 0:
            y
        else:
            x

    foo(1, 2)
    # => 1

    foo(0, 2)
    # => 2

    pvector(map(-> $1 * 2, [1, 2, 3]))
    # => pvector([2, 4, 6])

Pipeline operator
~~~~~~~~~~~~~~~~~

.. code:: python

    add = -> $1 + $2
    2 |> add(10) |> add(12)
    # => 24
    None |>? add(10) |>? add(12)
    # => None

Including a file at compile time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    $ cat anko.mochi
    x = 10000
    y = 20000

.. code:: python

    require 'anko.mochi'
    x
    # => 10000

    x = 30000

    require 'anko.mochi' # include once at compile time
    x
    # => 30000

Module
~~~~~~

.. code:: python

    module Math:
        export add, sub
        
        def add(x, y):
            x + y
        
        def sub(x, y):
            x - y

    Math.add(1, 2)
    # => 3

.. code:: sh

    $ cat foobar.mochi
    foo = 'foo'
    bar = 'bar'

.. code:: python

    require 'foobar.mochi'
    [foo, bar]
    # => pvector(['foo', 'bar'])

    foo = 'foofoofoo'

    module X:
        export foobar
        require 'foobar.mochi'
        def foobar:
            [foo, bar]

    X.foobar()
    # => pvector(['foo', 'bar'])

    [foo, bar]
    # => pvector(['foofoofoo', 'bar'])

TODO
----

-  Documentation
-  Improvement of parsing
-  Support class definition

License
-------

MIT License

Author
------

[i2y] (https://github.com/i2y)