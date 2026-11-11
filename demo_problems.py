"""Readable, curated summaries of published demo tasks; no hidden scoring cases."""

PROBLEMS = {
    "Cup swap": {
        "benchmark": "runbugrun",
        "id": "21346",
        "hint": "Compare M1 SFT with DPO or GRPO on this state-update bug.",
        "description": """## Cup swap

A ball starts under the cup at **A**. The three cup positions are **A**, **B**, and **C**. Each operation swaps two positions; the ball moves with its cup.

### Your task
Read all swaps and print the ball's final position.

### Input
One swap per line, with the two positions separated by a comma. Read until **end of input**; there is no leading count. There are at most **50 swaps**.

### Output
Print one letter: `A`, `B`, or `C`.

### Published example
**Input**
```text
B,C
A,C
C,B
A,B
C,B
```
**Output**
```text
A
```""",
    },
    "Remove a debug print": {
        "benchmark": "runbugrun",
        "id": "186967",
        "hint": "Compare SFT / GRPO with DPO. Watch the output contract.",
        "description": r"""## Count integer-product pairs

You receive positive decimal numbers $A_1, A_2, \ldots, A_N$.

### Your task
Count index pairs $(i,j)$ for which $i \lt j$ and $A_i A_j$ is an **integer**.

### Constraints
- $2 \le N \le 200{,}000$.
- $0 \lt A_i \lt 10^4$.
- Each value has at most **9 digits after the decimal point**.

### Input
The first line contains `N`. Each of the following `N` lines contains one decimal value.

### Output
Print only the number of qualifying pairs.

### Published example
**Input**
```text
5
7.5
2.4
17.000000001
17
16.000000000
```
**Output**
```text
3
```
The qualifying products are $7.5\times2.4$, $7.5\times16$, and $17\times16$.""",
    },
    "Largest proper divisor": {
        "benchmark": "humanevalpack",
        "id": "Python/24",
        "hint": "Compare pretrained with Instruct. Try DPO with and without feedback.",
        "description": """## Largest proper divisor

Implement this function:
```python
def largest_divisor(n: int) -> int:
    ...
```

### Your task
Return the **largest positive divisor of `n` that is smaller than `n`**.

### Interface
This is a function task: return the integer; do not read stdin or print an answer.

### Published example
```python
largest_divisor(15) == 5
```
The positive proper divisors of 15 are 1, 3, and 5.""",
    },
    "Run-length decoding": {
        "benchmark": "runbugrun",
        "id": "24076",
        "hint": "A short text-processing repair. Compare GRPO with SFT.",
        "description": """## Run-length decoding

Compressed text uses markers of the form **`@dX`**, meaning repeat character **`X`** exactly **`d`** times. Other characters are copied unchanged.

### Your task
Decode every input line.

### Input
Read compressed strings until end of input, one per line. There are at most **50 strings**. Repeat counts use one digit, up to **9**. A decoded string has at most **100 characters** and contains no `@`.

### Output
Print exactly one decoded string per input line.

### Published example
**Input**
```text
ab@5C1@8050
@99+1=1@90
```
**Output**
```text
abCCCCC10000000050
999999999+1=1000000000
```""",
    },
    "Arrange the cakes": {
        "benchmark": "runbugrun",
        "id": "858256",
        "hint": "A small runtime-error repair. Compare GRPO with M1 SFT.",
        "description": r"""## Arrange the cakes

There are $K$ cakes in $T$ types. Type $i$ has $a_i$ cakes. Eat one cake each day until all are gone.

### Your task
Choose the order to minimize the number of days whose cake has the **same type as the previous day's cake**. Print that minimum.

### Constraints
- $1 \le K \le 10{,}000$; $1 \le T \le 100$.
- $1 \le a_i \le 100$.
- $a_1 + \cdots + a_T = K$.

### Input
First line: `K T`. Second line: the `T` counts.

### Output
One integer: the minimum number of equal-type adjacencies.

### Published example
**Input**
```text
7 3
3 2 2
```
**Output**
```text
0
```
The types can be interleaved so no two consecutive days use the same type.""",
    },
    "Inclusive number sequence": {
        "benchmark": "humanevalpack",
        "id": "Python/15",
        "hint": "A compact boundary-condition example. Compare models on the same input.",
        "description": """## Inclusive number sequence

Implement this function:
```python
def string_sequence(n: int) -> str:
    ...
```

### Your task
Return the numbers **from 0 through `n`, inclusive**, as a string separated by single spaces.

### Interface
Return the string. Do not read input or print it.

### Published examples
```python
string_sequence(0) == "0"
string_sequence(5) == "0 1 2 3 4 5"
```""",
    },
}
