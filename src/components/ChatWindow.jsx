import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader } from 'lucide-react';

// Main Chat Component
const ChatWindow = () => {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'สวัสดีครับ เราคือเพื่อนใจวัยเรียน AI มีอะไรให้ช่วยรับฟังไหมครับ' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Effect to scroll to the bottom of the message list when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Function to handle sending a message
  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // API call to the backend
      const response = await fetch('http://localhost:8000/chat', { // Assuming backend runs on port 8000
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_prompt: input }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const aiMessage = { sender: 'ai', text: data.final_answer };
      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error("Failed to fetch from backend:", error);
      const errorMessage = { sender: 'ai', text: 'ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับระบบ' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle 'Enter' key press
  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-900 text-white font-sans">
      <div className="flex-grow container mx-auto p-4 flex flex-col max-w-3xl">
        <header className="text-center mb-4">
          <h1 className="text-2xl font-bold text-slate-300">เพื่อนใจวัยเรียน AI</h1>
          <p className="text-sm text-slate-500">พื้นที่ปลอดภัยสำหรับรับฟังและให้คำแนะนำเบื้องต้น</p>
        </header>
        
        {/* Message Display Area */}
        <div className="flex-grow bg-slate-800 rounded-xl shadow-inner p-4 overflow-y-auto space-y-4">
          {messages.map((msg, index) => (
            <div key={index} className={`flex items-end gap-2 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.sender === 'ai' && <Bot className="w-8 h-8 p-1.5 bg-slate-700 rounded-full flex-shrink-0" />}
              <div className={`px-4 py-2 rounded-2xl max-w-md md:max-w-lg ${msg.sender === 'user' ? 'bg-blue-600 rounded-br-none' : 'bg-slate-700 rounded-bl-none'}`}>
                <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
              </div>
              {msg.sender === 'user' && <User className="w-8 h-8 p-1.5 bg-slate-700 rounded-full flex-shrink-0" />}
            </div>
          ))}
          {isLoading && (
            <div className="flex items-end gap-2 justify-start">
               <Bot className="w-8 h-8 p-1.5 bg-slate-700 rounded-full flex-shrink-0" />
               <div className="px-4 py-2 rounded-2xl bg-slate-700 rounded-bl-none">
                 <Loader className="w-5 h-5 animate-spin text-slate-400" />
               </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        {/* Input Area */}
        <div className="mt-4 flex items-center gap-2 p-2 bg-slate-800 rounded-xl shadow-inner">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="พิมพ์ข้อความของคุณที่นี่..."
            className="w-full bg-slate-700 text-slate-200 placeholder-slate-500 rounded-lg px-4 py-2 border-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:opacity-50 text-white rounded-lg p-2 transition-colors duration-300 flex-shrink-0"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;

