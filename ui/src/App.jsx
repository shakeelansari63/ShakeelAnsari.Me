import { useState, useEffect } from 'react';

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetch('/api/hello')
      .then((res) => res.json())
      .then((data) => setMessage(data.message))
      .catch(() => setMessage('Hello, World!'));
  }, []);

  return (
    <div className="app">
      <h1>shakeelansari.me</h1>
      <p>{message}</p>
    </div>
  );
}

export default App;
