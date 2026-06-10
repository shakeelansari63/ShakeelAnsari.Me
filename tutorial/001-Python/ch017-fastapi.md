# FastAPI — Modern Web Framework

FastAPI is a modern, fast web framework for building APIs. Install with `pip install fastapi uvicorn`.

## Basic Setup

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root() -> dict[str, str]:
    return {"Hello": "World"}

@app.get("/about")
def about() -> str:
    return "This is great work"
```

Run the server:

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs` for interactive API docs.

## Route Decorators

```python
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

## Type Hints for Response Models

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

@app.post("/items/")
def create_item(item: Item) -> Item:
    return item

@app.get("/items/")
def list_items() -> list[Item]:
    return [Item(name="Apple", price=1.99)]
```

FastAPI uses Python type hints to automatically validate request data, serialize responses, and generate OpenAPI documentation.
