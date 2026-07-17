# Step-1 dyn-τ blind quality scoring (2026-07-17)

Score each output 1–5 (5 best) for: coherence (C), instruction adherence (I), factual consistency where applicable (F, else '-'). Then overall verdict A/B/tie. Do NOT open key.json first. The two arms behind A/B are shuffled per prompt and withheld — see README.md for what the sheet decides (it does not reveal which is which).


## code-fizzbuzz

**Prompt**: Write a Python function fizzbuzz(n) that returns a list of the classic FizzBuzz strings for 1..n.


### Output A

```
Here's a Python function that implements the classic FizzBuzz problem:

```python
def fizzbuzz(n):
    """
    Returns a list of FizzBuzz strings for numbers 1 to n.
    
    For numbers divisible by 3: "Fizz"
    For numbers divisible by 5: "Buzz"
    For numbers divisible by both 3 and 5: "FizzBuzz"
    For all other numbers: the number as a string
    
    Args:
        n (int): The upper limit (inclusive)
    
    Returns:
```

C: __  I: __  F: __


### Output B

```
Here's a Python function that implements the classic FizzBuzz problem:

```python
def fizzbuzz(n):
    """
    Returns a list of FizzBuzz strings for numbers 1 to n.
    
    For numbers divisible by 3: "Fizz"
    For numbers divisible by 5: "Buzz"
    For numbers divisible by both 3 and 5: "FizzBuzz"
    For all other numbers: the number as a string
    
    Args:
        n (int): The upper limit (inclusive)
    
    Returns:
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## code-regex

**Prompt**: Write a regular expression that matches ISO-8601 dates (YYYY-MM-DD) and explain each part briefly.


### Output A

```
Here's a regular expression that matches ISO-8601 dates in the format `YYYY-MM-DD`:

```regex
^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$
```

---

### 🔍 Explanation of Each Part:

1. **`^`**  
   - **Start of string anchor**: Ensures the match starts at the beginning of the string.

2. **`\d{4}`**  
   -
```

C: __  I: __  F: __


### Output B

```
Here's a regular expression that matches ISO-8601 dates in the format `YYYY-MM-DD`:

```regex
^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$
```

---

### 🔍 Explanation of Each Part:

1. **`^`**  
   - **Start of string anchor**: Ensures the match starts at the beginning of the string.

2. **`\d{4}`**  
   -
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## code-sql

**Prompt**: Given a table orders(id, customer_id, total, created_at), write a SQL query for the top 5 customers by total spend in 2025.


### Output A

```
Here's a SQL query to find the top 5 customers by total spend in 2025:

```sql
SELECT 
    customer_id,
    SUM(total) AS total_spend
FROM orders
WHERE YEAR(created_at) = 2025
GROUP BY customer_id
ORDER BY total_spend DESC
LIMIT 5;
```

**Alternative version using different date functions:**

```sql
SELECT 
    customer_id,
    SUM(total) AS total_spend
FROM orders
WHERE created_at >= '2025-01-01
```

C: __  I: __  F: __


### Output B

```
Here's a SQL query to find the top 5 customers by total spend in 2025:

```sql
SELECT 
    customer_id,
    SUM(total) AS total_spend
FROM orders
WHERE YEAR(created_at) = 2025
GROUP BY customer_id
ORDER BY total_spend DESC
LIMIT 5;
```

**Alternative version using explicit date range:**

```sql
SELECT 
    customer_id,
    SUM(total) AS total_spend
FROM orders
WHERE created_at >= '2025-01-01
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## code-swift-struct

**Prompt**: Define a Swift struct Point with x and y Doubles and a method distance(to:) returning the Euclidean distance.


### Output A

```
Here's a Swift struct `Point` with `x` and `y` properties and a `distance(to:)` method:

```swift
struct Point {
    var x: Double
    var y: Double
    
    func distance(to point: Point) -> Double {
        let dx = self.x - point.x
        let dy = self.y - point.y
        return sqrt(dx * dx + dy * dy)
    }
}

// Example usage:
let point1 = Point(x: 0.0, y: 0.0)
let point2
```

C: __  I: __  F: __


### Output B

```
Here's a Swift struct `Point` with `x` and `y` properties and a `distance(to:)` method:

```swift
struct Point {
    var x: Double
    var y: Double
    
    func distance(to other: Point) -> Double {
        let dx = other.x -.x
        let dy = other.y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

// Example usage:
let point1 = Point(x: 0.0, y: 0.0)
let point2 =
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## reason-ages

**Prompt**: Anna is twice as old as Ben was when Anna was as old as Ben is now. Anna is 24 and Ben is 18. Check whether this statement is consistent and explain.


### Output A

```
We are given:

- Anna is **24 years old**
- Ben is **18 years old**
-
 statement statement:



AnnaAnna is as old as Ben was when Anna was as old as Ben is now."

We are to check whether this statement is **consistent**.

---

### Step 1: Understand the statement

Let’s break down the statement:

> "Anna is twice as old as Ben was when Anna was as old as Ben is now."

Let’s define:

- Anna's current age: 24

```

C: __  I: __  F: __


### Output B

```
We are given:

- Anna is **24 years old**
- Ben is **18 years old**
- The statement: *"Anna is twice as old as Ben was when Anna was as old as Ben is now."*

We need to check whether this statement is **consistent**.

---

### Step 1: Understand the statement

Let’s break down the sentence:

> "Anna is twice as old as Ben was when Anna was as old as Ben is now."

Let’s define:

- Anna’s current age: 24
- Ben
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## reason-logic

**Prompt**: All roses are flowers. Some flowers fade quickly. Does it follow that some roses fade quickly? Answer yes or no and justify in two sentences.


### Output A

```
No. The fact that all roses are flowers and that some flowers fade quickly does not necessarily that that some roses fade quickly, because the flowers that fade quickly might not be roses at all.<|role_end|><|endoftext|>
```

C: __  I: __  F: __


### Output B

```
No. The fact that all roses are flowers and that some flowers fade quickly does not necessarily that that some roses fade quickly, because the flowers that fade quickly might not include any roses at all.<|role_end|><|endoftext|>
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## reason-trains

**Prompt**: Two trains are 240 km apart and drive toward each other, one at 70 km/h and one at 50 km/h. After how many hours do they meet? Show your reasoning briefly.


### Output A

```
We are given:

- Distance between the two trains: 240 km  
- Speed of first train: 70 km/h  
- Speed of second train: 50 km/h  
- They are moving **toward each other**

### Step 1: Add their speeds
Since they are moving toward each other, their combined speed is:

$$
70\,\text{km/h} + 50\,\text{km/h} = 120\,\text{km/h}
$$

### Step 2: Use the formula time


```

C: __  I: __  F: __


### Output B

```
We are given:

- Distance between the two trains: 240 km  
- Speed of first train: 70 km/h  
- Speed of second train: 50 km/h  
- They are moving **toward each other**

### Step 1: Relative speed
When two objects move toward each other, their relative speed is the sum of their speeds:

$$
70 \text{ km/h} + 50 \text{ km/h} = 120 \text{ km/h}
$$

### Step 2: Time to
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __


## reason-units

**Prompt**: A recipe needs 3/4 cup of milk. I only have a 1/3-cup measure. How many times must I fill it, and how much do I overshoot or undershoot?


### Output A

```
We need **3/4 cup** of milk and you only have a **1/3-cup** measure.

### Step 1: Determine how many times you need to fill the 1/3-cup measure

We divide:

$$
\frac{3}{4} \div \frac{1}{3} = \frac{3}{4} \times \frac{3}{1} = \frac{9}{4} = 2.25
$$

So, you need to fill the 1/3-cup measure **2.25 times**.

```

C: __  I: __  F: __


### Output B

```
We need **3/4 cup** of milk and you only have a **1/3-cup** measure.

### Step 1: Determine how many 1/3-cup measures fit into 3/4 cup

We compute:

$$
\frac{3/4}{1/3} = \frac{3}{4} \times \frac{3}{1} = \frac{9}{4} = 2.25
$$

So, you need to fill the 1/3-cup measure **2.25 times**.

### Step 
```

C: __  I: __  F: __


**Overall (A/B/tie)**: __
