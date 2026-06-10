# Arrays & Slices

## Arrays

Arrays have a **fixed length** determined at compile time. They are **value types** — assigning an array to another variable copies the entire contents.

```go
var arr1 [3]string
arr1[0] = "Hello"; arr1[1] = "Hi"; arr1[2] = "Bye"

arr2 := [3]int{10, 11, 15}
arr3 := [...]float32{10.5, 12.8, 3.14}  // compiler counts the elements
```

The `...` syntax lets the compiler infer the array size from the initializer.

```go
fmt.Println(arr3[1])        // Index access: 12.8
fmt.Println(len(arr1))      // Length: 3
```

### Multi-Dimensional Arrays

```go
var arr4 [3][3]int
arr4[0] = [3]int{1, 0, 0}
arr4[1] = [3]int{0, 1, 0}
arr4[2] = [3]int{0, 0, 1}
```

### Value Semantics

Arrays are actual values, not pointers. When you copy an array, you get a completely independent copy:

```go
arr6 := arr3
arr6[2] = 6.9               // arr3 is unchanged
```

This means passing a large array to a function copies the entire array, which can be a performance concern. The workaround is to use a pointer:

```go
arr7 := &arr6
arr7[2] = 11.0              // now arr6 also changes
```

## Slices

Slices are **dynamic-length** views into arrays. They are **reference types** — copying a slice shares the underlying data.

```go
slc1 := []int{1, 2, 3}     // No size specified — this is a slice, not an array
slc2 := slc1
slc2[2] = 5                 // slc1 also changes (same backing array)
```

### Slicing

The slice syntax `[low:high]` creates a new slice that references the same underlying array:

```go
slc3 := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
slc4 := slc3[:]             // all elements
slc5 := slc3[4:]            // index 4 to end
slc6 := slc3[:5]            // start to index 4
slc7 := slc3[2:7]           // index 2 to 6
```

### Length vs Capacity

- `len(slice)` — number of elements currently in the slice
- `cap(slice)` — maximum elements the underlying array can hold from the slice's start position

For `slc7` (elements [2..7) of `slc3`), the length is 5 but the capacity extends to the end of `slc3`:

```go
fmt.Println(len(slc7), cap(slc7))  // 5, 8
```

### Make with Capacity

`make` creates a slice with a specified length and capacity:

```go
slc8 := make([]int, 3, 100)  // type, length, capacity
```

### Append

The `append` function adds elements to a slice. If the backing array has capacity, it reuses it; otherwise, Go allocates a new, larger array.

```go
slc9 := []int{}
slc9 = append(slc9, 5)           // add one element
slc9 = append(slc9, 2, 6, 3, 8, 7)  // add multiple elements
slc9 = append(slc9, slc6...)     // spread another slice with ...
```

### Remove Elements

Go has no built-in remove function, but slicing achieves the same result:

```go
slc9 = slc9[1:]                 // remove first element
slc9 = slc9[:len(slc9)-1]       // remove last element
slc9 = append(slc9[:n], slc9[n+1:]...)  // remove nth element
```
