# Arrays

Arrays store multiple values of the same type in a fixed-size sequential collection.

## Declaring and Initializing

```java
// Method 1: Declare then assign
int[] numbers = new int[5];
numbers[0] = 10;
numbers[1] = 20;

// Method 2: Inline initialization
int[] scores = {95, 87, 92, 78, 88};

// Method 3: Using new with values
String[] names = new String[]{"Alice", "Bob", "Charlie"};
```

## Accessing Elements

```java
int[] arr = {10, 20, 30, 40, 50};
arr[0];                // 10
arr[arr.length - 1];   // 50 (last element)
arr[5];                // ArrayIndexOutOfBoundsException!
```

## Array Length

```java
int[] arr = {1, 2, 3};
arr.length;   // 3 (note: no parentheses — it's a field, not a method)
```

## Iterating

```java
int[] scores = {95, 87, 92};

// Traditional for loop
for (int i = 0; i < scores.length; i++) {
    System.out.println(scores[i]);
}

// Enhanced for-each loop
for (int score : scores) {
    System.out.println(score);
}
```

## Multi-Dimensional Arrays

```java
int[][] matrix = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

matrix[0][1];  // 2
matrix[2][2];  // 9

// Nested loop
for (int[] row : matrix) {
    for (int val : row) {
        System.out.print(val + " ");
    }
    System.out.println();
}
```

## Array Utility Methods

```java
import java.util.Arrays;

int[] arr = {5, 2, 8, 1, 9};
Arrays.sort(arr);                          // [1, 2, 5, 8, 9]
Arrays.binarySearch(arr, 5);               // 2 (index)
Arrays.toString(arr);                      // "[1, 2, 5, 8, 9]"
Arrays.fill(arr, 0);                       // [0, 0, 0, 0, 0]
int[] copy = Arrays.copyOf(arr, arr.length);
```
