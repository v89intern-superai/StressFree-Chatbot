import React from 'react';
import ChatWindow from './components/ChatWindow'; // <-- เรียกนักแสดงหลักของเรามา

function App() {
  return (
    // จัดฉากให้ทุกอย่างอยู่ตรงกลางจอ
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <ChatWindow />
    </div>
  );
}

export default App;