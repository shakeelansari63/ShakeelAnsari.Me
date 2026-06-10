# Socket Programming — Client & Server

Python's `socket` module lets you create networked applications. This example uses TCP with threading.

## Server

The server listens for connections and handles each client in a separate thread.

```python
import socket
import threading

PORT = 5050
HOST = socket.gethostbyname(socket.gethostname())
ADDR = (HOST, PORT)
HEADER_SIZE = 64
DATA_FORMAT = 'utf-8'
DISCONNECT_MSG = '!DISCONNECT'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

def handle_client(conn, addr):
    print(f'[NEW CONNECTION] : {addr} connected.')
    connected = True
    while connected:
        msg_len = conn.recv(HEADER_SIZE).decode(DATA_FORMAT)
        if msg_len:
            msg = conn.recv(int(msg_len)).decode(DATA_FORMAT)
            print(f'[ {HOST}:{PORT} ] : {msg}')
            if msg == DISCONNECT_MSG:
                connected = False
            else:
                conn.send('Message Received !!!'.encode(DATA_FORMAT))
    conn.close()

def start():
    server.listen()
    print(f'[LISTENING] Server listening at {HOST}:{PORT}')
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

if __name__ == '__main__':
    try:
        start()
    except KeyboardInterrupt:
        server.close()
```

## Client

The client connects to the server, sends a header with message length, then the message.

```python
import socket

PORT = 5050
HOST = '127.0.1.1'
ADDR = (HOST, PORT)
HEADER_SIZE = 64
DATA_FORMAT = 'utf-8'
DISCONNECT_MSG = '!DISCONNECT'

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send_msg(msg):
    message = msg.encode(DATA_FORMAT)
    msg_len = str(len(message)).encode(DATA_FORMAT)
    send_len = msg_len + b' ' * (HEADER_SIZE - len(msg_len))
    client.send(send_len)
    client.send(message)
    print(f'[SERVER MESSAGE]: {client.recv(2048).decode(DATA_FORMAT)}')

if __name__ == '__main__':
    send_msg('Hello, World !!!')
    send_msg(DISCONNECT_MSG)
```

## How It Works

1. The server binds to an address and listens for connections.
2. When a client connects, the server creates a new thread to handle it.
3. Each message is preceded by a 64-byte header containing the message length.
4. The client sends the header, then the actual message.
5. The server reads the header, then reads the message of that length.
6. Sending `!DISCONNECT` closes the connection cleanly.
