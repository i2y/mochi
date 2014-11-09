Mochi
====

Mochi is a dynamically typed functional programming language.


## Summary
Mochi is a dynamically typed functional programming language.
Its interpreter is written in Python3.
A program written in Mochi is compiled to Python's AST / bytecode by the interpreter.



## Features
- Python-like syntax
- Tail recursion optimization (self tail recursive only), and no loop syntax
- Re-assignment are not allowed in the function definition.
- Basic data type is a persistent data structure (using Pyrsistent)
- Pattern matching / Algebraic data type
- Pipeline operator
- Syntax sugar of anonymous function definition
- Built-in Python3 itertools and functools, operator module functions and function in itertools recipes


## Dependencies
- CPython >= 3.2 or PyPy >= 3.2.1
- rply >= 0.7.2
- pyrsistent >= 0.6.2
- pathlib >= 1.0.1


## Installation
```sh
$ pip install git+https://github.com/i2y/mochi.git
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
$ kinako.mochi mochi
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

## Examples


### Persistent data structure
```python
[1, 2, 3]
# => pvector([1, 2, 3])

v(1, 2, 3)
# => pvector([1, 2, 3])

{'x': 100, 'y': 200}
# => pmap({'y': 200, 'x': 100})

m(x=100, y=200)
# => pmap({'y': 200, 'x': 100})

s(1, 2, 3)
# => pset([1, 2, 3])

b(1, 2, 3)
# => pbag([1, 2, 3])
```

### Function definition
```python
def hoge(x):
    hoge + str(x)

hoge(3)
# => hoge3
```

### Pattern matching
```python
lis = [1, 2, 3]
match lis:
    [1, 2, x]: x
    _: None
# => 3

match lis:
    [1, &rest]: rest
    _: None

# => pvector (2, 3)
```

### Algebraic data type
```python
aata Point:
    Point2D(x, y)
    Point3D(x, y, z)

p1 = Point2D(x=1, y=2)
# => Point2D(x=1, y=2)

p2 = Point2D(3, 4)
# => Point2D(x=3, y=4)
```

### Pattern-matching function definition
```python
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
# => Point2D(x=3, y=4)
offset(Point3D(1, 2, 3), Point3D(4, 5, 6))
# => Point3D(x=5, y=7, z=9)
```

### Pipeline operator 
```python
def fizzbuzz(n):
    match [n % 3, n % 5]:
        [0, 0]: "fizzbuzz"
        [0, _]: "fizz"
        [_, 0]: "buzz"
        _: n

range(1, 31) |> map(fizzbuzz) |> pvector() |> print()
# => pvector([1, 2, 'fizz', 4, 'buzz', 'fizz', 7, 8, 'fizz', 'buzz', 11, 'fizz', 13, 14, 'fizzbuzz', 16, 17, 'fizz', 19, 'buzz', 'fizz', 22, 23, 'fizz', 'buzz', 26, 'fizz', 28, 29, 'fizzbuzz'])
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

pvector(map(-> $ 1 * 2, [1, 2, 3]))
# => pvector([2, 4, 6])
```


## TODO
- Documentation
- Improvement of parsing
- Support class definition

## License
MIT License

## Author
[i2y] (https://github.com/i2y)