# Flask — Web Framework

Flask is a lightweight web framework. Install with `pip install flask`.

## Basic Setup

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
```

Run with:
```bash
export FLASK_APP=app.py
flask run
```

Or with debug mode:
```bash
export FLASK_APP=app.py
flask --debug run
```

## Routes

```python
@app.route('/')
def index():
    return "<h1>Home Page</h1>"

@app.route('/about')
def about():
    return "<h1>About Page</h1>"

@app.route('/user/<string:name>')
def user(name):
    return f"<h1>Hello, {name}!</h1>"
```

## Templates

Create a `templates/` folder with HTML files.

```python
from flask import render_template

@app.route('/')
def index():
    return render_template('index.html', name="Alice")
```

`templates/index.html`:
```html
<h1>Hello, {{ name }}!</h1>
```

## SQLAlchemy ORM

Install `flask-sqlalchemy`.

```python
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

class ToDo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Integer, default=0)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Task {self.id}>'
```

## CRUD Todo App

```python
@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        task = ToDo(content=task_content)
        try:
            db.session.add(task)
            db.session.commit()
            return redirect('/')
        except:
            return 'Error occurred'
    else:
        tasks = ToDo.query.order_by(ToDo.date_created).all()
        return render_template('index.html', tasks=tasks)

@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = ToDo.query.get_or_404(id)
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'Error Occurred'

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task_to_update = ToDo.query.get_or_404(id)
    if request.method == 'GET':
        return render_template('update.html', task=task_to_update)
    else:
        task_to_update.content = request.form['content']
        try:
            db.session.commit()
            return redirect('/')
        except:
            return "Error Occurred"
```

## JWT Authentication

Install `pip install pyjwt`.

```python
from flask import Flask, jsonify, make_response, request
import jwt
import datetime as dt
from functools import wraps

app = Flask(__name__)
app.config['secret'] = 'MySecretKeyForApp'

def token_required(f):
    @wraps(f)
    def pre_processor(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'message': 'Token missing!!'}), 401
        try:
            jwt.decode(token, app.config['secret'], algorithms=["HS256"])
        except:
            return jsonify({'message': 'Invalid Token!!'}), 401
        return f(*args, **kwargs)
    return pre_processor

@app.route('/public')
def public_page():
    return jsonify({'message': 'Public data'}), 200

@app.route('/private')
@token_required
def private_page():
    return jsonify({'message': 'Private message for authenticated users'}), 200

@app.route('/login')
def login():
    auth = request.authorization
    if auth and auth.password == 'secret':
        token = jwt.encode({'user': auth.username,
                            'exp': dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30)},
                           app.config['secret'], algorithm="HS256")
        return jsonify({'token': token})
    return make_response('Could not Authenticate!', 401,
                         {'WWW-Authenticate': 'Basic realm="Login required"'})
```
