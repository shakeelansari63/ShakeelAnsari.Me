# Type Testing & Casting

Scala provides two operators for runtime type inspection and casting: `isInstanceOf` and `asInstanceOf`.

## isInstanceOf — Type Checking

`isInstanceOf` tells whether an object is an instance of a given type at runtime:

```scala
println(3.isInstanceOf[Int])           // true
println(3.isInstanceOf[Double])        // false
println("3".isInstanceOf[String])       // true
println("3".isInstanceOf[CharSequence]) // true (String implements CharSequence)
```

The check works with the full type hierarchy — `"3"` is a `String`, but it's also a `CharSequence` because `String` implements that interface.

## asInstanceOf — Type Casting

`asInstanceOf` casts an object from one type to another type in the same inheritance chain:

```scala
println(3.asInstanceOf[Double]) // 3.0
```

Casting does not convert the value — it reinterprets the reference type. Use it only when you are certain the object is of the target type, or you risk a `ClassCastException`.

## Using Them Together

A common pattern is to check the type with `isInstanceOf` first, then cast with `asInstanceOf`:

```scala
def decide(a: Any) = if (a.isInstanceOf[Int]) a.asInstanceOf[Int] + 1
else -1

println(decide(4))       // 5
println(decide("Hello")) // -1
```

Here `a` is typed as `Any`, so we must check and cast before using it as an `Int`. This is the runtime equivalent of pattern matching (which is the more idiomatic Scala approach).
