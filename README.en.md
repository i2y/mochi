Mochi
====
Mochi is a dynamically typed programming language for functional programming and actor-style programming.

Its interpreter is written in Python3. The interpreter translates a program written in Mochi to Python3's AST / bytecode.

## Features
- Python-like syntax
- Tail recursion optimization (self tail recursion only), and no loop syntax
- Re-assignments are not allowed in function definition.
- Basic collection type is a persistent data structure. (using Pyrsistent)
- Pattern matching / Data types, like algebraic data types
- Pipeline operator
- Syntax sugar of anonymous function definition
- Actor, like the actor of Erlang (using Eventlet)
- Macro, like the traditional macro of Lisp
- Builtin functions includes functions exported by itertools module, recipes, functools module and operator module


## Examples
### Factorial
```python
def factorial(n, m):
    if n == 1:
        m
    else:
        factorial(n - 1, n * m)


factorial(10000, 1)
# => 28462596809170545189064132121198688...

# Or

def factorial:
    n: factorial(n, 1)
    0, acc: acc
    n, acc: factorial(n - 1, acc * n)
    
factorial(10000)
# => 28462596809170545189064132121198688...
```

### FizzBuzz
```python
def fizzbuzz(n):
    match [n % 3, n % 5]:
        [0, 0]: "fizzbuzz"
        [0, _]: "fizz"
        [_, 0]: "buzz"
        _: n

range(1, 31)
|> map(fizzbuzz)
|> pvector()
|> print()
```

### Actor
```python
def show():
    receive:
        message:
            print(message)
            show()

actor = spawn(show)

send('foo', actor)
actor ! 'bar' # send('bar', actor)

sleep(1)
# -> foo
# -> bar


'foo' !> spawn(show)

sleep(1)
# -> foo

['foo', 'bar'] !&> spawn(show)
# The meaning of the above is the same as the meaning of the following.
# spawn(show) ! 'foo'
# spawn(show) ! 'bar'

sleep(1)
# -> foo
# -> bar

def show_loop():
    receive:
        [tag, value]:
            print(tag, value)
            show_loop()

actor2 = spawn(show_loop)

actor2 ! ["bar", 2000]
sleep(1)
# -> bar 2000

['foo', 1000] !> spawn(show_loop)
sleep(1)
# -> foo 1000

[['foo', 1000],['bar', 2000]] !&> spawn(show_loop)
sleep(1)
# -> foo 1000
# -> bar 2000

remote_actor = RemoteActor('tcp://localhost:9999/test')
remote_actor ! ['remote!', 3000]

hub = ActorHub('tcp://*:9999')
hub.register('test', actor2)
hub.run()

wait_all()
# -> remote! 3000
```

### Flask
```python
from flask import Flask

app = Flask('demo')

@app.route('/')
def hello():
    'Hello World!'

app.run()
```

### aif (Anaphoric macro)
```python
macro aif(test, true_expr, false_expr):
    quasi_quote:
        it = unquote(test)
        if it:
            unquote(true_expr)
        else:
            unquote(false_expr)

aif([], first(it), "empty")
# => "empty"
aif([10, 20], first(it), "empty")
# => 10
```

## Requirements
- CPython >= 3.2 or PyPy >= 3.2.1
- rply >= 0.7.2
- pyrsistent >= 0.9.1
- pathlib >= 1.0.1
- eventlet >= 0.17.1
- pyzmq >= 14.5.0
- msgpack-python >= 0.4.6
- kazoo >= 2.0
- typeannotations >= 0.1.0


## Installation
```sh
$ pip3 install mochi
$ pip3 install flask Flask-RESTful Pillow  # to run the examples
```

Th error of the following may occur when you run the mochi on PyPy.
```
ImportError: Importing zmq.backend.cffi failed with version mismatch, 0.8.2 != 0.9.2
```
In this case, please change the version of cffi to 0.8.2 using pip on PyPy.
```sh
$ pip3 uninstall cffi
$ pip3 install cffi==0.8.2
```

## Usage

### REPL
```sh
$ mochi
>>>
```

### loading and running a file
```sh
$ cat kinako.mochi
print('kinako')
$ mochi kinako.mochi
kinako
$
```

### byte compilation
```sh
$ mochi -c kinako.mochi > kinako.mochic
```

### running a byte-compiled file
```sh
$ mochi -e kinako.mochic
kinako
$
```

### generating .pyc
```sh
$ cat kagami.mochi
print('kagami')
$ mochi -pyc kagami.mochi > kagami.pyc
$ python3 kagami.pyc
kagami
$ python3
>>> import kagami
kagami
>>> eixt()
$
```

## Examples for each feature

### Persistent data structures
```python
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

get(vec, 0) # => 1
get(vec, 0, 2) # => [1, 2]

vec[0] # => 1
vec[0:2] # => [1, 2]

{'x': 100, 'y': 200}
# => pmap({'y': 200, 'x': 100})

ma = {'x': 100, 'y': 200}
ma.get('x') # => 100
ma.x # => 100
ma['x'] # => 100
ma2 = ma.set('x', 10000)
# => pmap({'y': 200, 'x': 10000})
ma # => pmap({'y': 200, 'x': 100})
get(ma, 'y') # => 200
ma['y'] # => 200

m(x=100, y=200)
# => pmap({'y': 200, 'x': 100})

s(1, 2, 3)
# => pset([1, 2, 3])

b(1, 2, 3)
# => pbag([1, 2, 3])
```

### Function definitions
```python
def hoge(x):
    'hoge' + str(x)

hoge(3)
# => hoge3
```

### Pattern matching
```python
lis = [1, 2, 3]

# Sequence pattern
match lis:
    [1, 2, x]: x
    _: None
# => 3

match lis:
    [1, &rest]: rest
    _: None

# => pvector([2, 3])


foo_map = {'foo' : 'bar'}

# Mapping pattern
match foo_map:
    {'foo' : value}: value
    _: None
# => 'bar'


# Type pattern
# <name of variable refers to type> <pattern>: <action>
match 10:
    int x: 'int'
    float x: 'float'
    str x: 'str'
    bool x: 'bool'
    _: 'other'
# => 'int'

match [1, 2, 3]:
    [1, str x, 3]: 'str'
    [1, int x, 3]: 'int'
    _: 'other'
# => 'int'

num = union(int, float)
vector nums[num]
vector strs[str]

match nums([1, 2, 3]):
    nums[x, y, z]: z
    strs[x, y, z]: x
# => 3

Positive = predicate(-> $1 > 0)
Even = predicate(-> $1 % 2 == 0)
EvenAndPositive = predicate(-> ($1 % 2 == 0) and ($1 >= 0)) 

match 10:
    EvenAndPositive n: str(n) + ':Even and Positive'
    Even n: str(n) + ':Even'
    Positive n: str(n) + ':Positive'

# => 10:Even and Positive


# Or pattern
match ['foo', 100]:
    ['foo' or 'bar', value]: value
    _: 10000
# => 100

match ['foo', 100]:
    [str x or int x, value]: value
    _: 10000
# => 100


# Record pattern
record Person(name, age)

foo = Person('foo', 32)

match foo:
    Person('bar', age):
        'bar:' + str(age)
    Person('foo', age):
        'foo:' + str(age)
    _: None
# => 'foo:32'
```

### Records
```python
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

# runtime type checking
record Point(x:int, y:int, z:optional(int))
Point(1, 2, None)
# => Point(x=1, y=2, z=None)
Point(1, 2, 3)
# => Point(x=1, y=2, z=3)
Point(1, None, 3)
# => TypeError
```

### Bindings
```python
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
```

### Data types, like algebraic data types
```python
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
```

### Pattern-matching function definitions
```python
data Point:
    Point2D(x, y)
    Point3D(x, y, z)

def offset:
    Point2D(x1, y1), Point2D(x2, y2):
        Point2D(x1 + x2, y1 + y2)
    Point3D(x1, y1, z1), Point3D(x2, y2, z2):
        Point3D(x1 + x2, y1 + y2, z1 + z2)
    _: None

offset(Point2D(1, 2), Point2D(3, 4))
# => Point2D(x=4, y=6)
offset(Point3D(1, 2, 3), Point3D(4, 5, 6))
# => Point3D(x=5, y=7, z=9)

def show:
    int x, message: print('int', x, message)
    float x, message: print('float', x, message)
    _: None

show(1.0, 'msg')
# -> float 1.0 msg
# => None

FileMode = options('r', 'w', 'a', 'r+', 'w+', 'a+')

def open_file:
    str path, FileMode mode: 
        open(path, mode)
    str path:
        open(path, 'r')
```

### Anonymous function
```python
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
```

### Pipeline operator 
```python
add = -> $1 + $2
2 |> add(10) |> add(12)
# => 24
None |>? add(10) |>? add(12)
# => None
```

### Lazy sequences
```python
def fizzbuzz(n):
    match [n % 3, n % 5]:
        [0, 0]: "fizzbuzz"
        [0, _]: "fizz"
        [_, 0]: "buzz"
        _: n


result = range(1, 31) |> map(fizzbuzz)
pvector(result)
# => pvector([1, 2, fizz, 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23, 'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz'])
pvector(result)
# => pvector([])
pvector(result)
# => pvector([])


# Iterator -> lazyseq
lazy_result = range(1, 31) |> map(fizzbuzz) |> lazyseq()
pvector(lazy_result)
# => pvector([1, 2, fizz, 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23, 'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz'])
pvector(lazy_result)
# => pvector([1, 2, fizz, 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23, 'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz'])
pvector(lazy_result)
# => pvector([1, 2, fizz, 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23, 'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz'])
```

### Trailing closures
```python
# The following trailing closure expression is passed to a function as the function’s first argument.
result = map([1, 2, 3]) ->
    print($1)
    $1 * 2

print(doall(result))

# -> 1
# -> 2
# -> 3
# => pvector([2, 4, 6])


def foreach(closure, seq):
    doall(filter(closure, seq))

# The following trailing closure expression is passed to a function as the function’s first argument.
foreach([1, 2, 3]) (item) ->
    new_item = item * 100
    print(new_item)

# -> 100
# -> 200
# -> 300
# => pvector([])

# Or

def foreach(seq, closure):
    doall(filter(closure, seq))

# The following trailing closure expression is passed to a function as the function’s final argument.
foreach([1, 2, 3]) @ (item) ->
    new_item = item * 100
    print(new_item)

# -> 100
# -> 200
# -> 300
# => pvector([])
```

### Short form for keyword arguments and dict keys
```python
def foo(a, b, c):
    a + b + c
    
a = 1
b = 2
c = 3

# This is the same as foo(a=a, b=b, c=c)
foo(=a, =b, =c))
# => 6

# This is the same as {'a': a, 'b': b}
{=a, =b}
# => pmap({'a': 1, 'b': 2})
```

### Macros
```python
macro rest_if_first_is_true(first, &args):
     match first:
         quote(True): quasi_quote(v(unquote_splicing(args)))
         _: quote(False)

rest_if_first_is_true(True, 1, 2, 3)
# => pvector([1, 2, 3])
rest_if_first_is_true("foo", 1, 2, 3)
# => False

macro pipeline(&args):
    [Symbol('|>')] + args

pipeline([1, 2, 3],
         map(-> $1 * 2),
         filter(-> $1 != 2),
         pvector())
# => pvector([4, 6])
```

### Including a file at compile time
```sh
$ cat anko.mochi
x = 10000
y = 20000
```

```python
require 'anko.mochi'
x
# => 10000

x = 30000

require 'anko.mochi' # include once at compile time
x
# => 30000
```

### Module
```python
module Math:
    export add, sub
    
    def add(x, y):
        x + y
    
    def sub(x, y):
        x - y

Math.add(1, 2)
# => 3
```

```sh
$ cat foobar.mochi
foo = 'foo'
bar = 'bar'
```

```python
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
```

## TODO
- Improve documentation
- Improve parsing
- Support type annotation

## License
MIT License

## Author
[i2y] (https://github.com/i2y)
