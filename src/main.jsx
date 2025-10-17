import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css' // <-- Import CSS หลัก (ที่มี Tailwind)

// นี่คือคำสั่งของ "ผู้กำกับ"
// 1. ไปหา "เวที" ที่ชื่อ 'root' ใน index.html
// 2. นำ "บทละคร" (<App />) ไปเริ่มจัดแสดงบนนั้น
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
