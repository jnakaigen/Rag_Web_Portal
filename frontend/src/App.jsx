import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  
  // New States for UI feedback
  const [chatLoading, setChatLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState(""); // "idle", "processing", "success", "error"
  const [statusMessage, setStatusMessage] = useState("");

  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  // 1. Send Message
  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: "user", text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setChatLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.text })
      });

      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail);

      const botMsg = { 
        role: "bot", 
        text: data.answer, 
        sources: data.sources 
      };
      setMessages(prev => [...prev, botMsg]);

    } catch (error) {
      setMessages(prev => [...prev, { role: "bot", text: "⚠️ Error: " + error.message }]);
    }
    setChatLoading(false);
  };

  // 2. Ingest URL
  const handleUrlSubmit = async () => {
    if (!url) return;
    setIngestStatus("processing");
    setStatusMessage("🌐 Scraping & Chunking Website...");

    try {
      const res = await fetch("http://localhost:8000/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail);

      setIngestStatus("success");
      setStatusMessage("✅ " + (data.message || "URL Ingested!"));
      setUrl("");
    } catch (err) {
      setIngestStatus("error");
      setStatusMessage("❌ Failed: " + err.message);
    }
    
    // Clear status after 5 seconds
    setTimeout(() => {
      if (ingestStatus !== "processing") {
        setIngestStatus("");
        setStatusMessage("");
      }
    }, 5000);
  };

  // 3. Ingest PDF
  const handleFileUpload = async () => {
    if (!file) return;
    setIngestStatus("processing");
    setStatusMessage("📄 Parsing PDF & Embedding...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/ingest/pdf", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail);

      setIngestStatus("success");
      setStatusMessage("✅ " + (data.message || "PDF Ingested!"));
      setFile(null);
    } catch (err) {
      setIngestStatus("error");
      setStatusMessage("❌ Failed: " + err.message);
    }

    setTimeout(() => {
      setIngestStatus("");
      setStatusMessage("");
    }, 5000);
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <h2>Knowledge Base</h2>
        
        {/* Status Notification Box */}
        {statusMessage && (
          <div className="status-indicator">
            {ingestStatus === "processing" ? <span className="loading-dots">Processing</span> : statusMessage}
          </div>
        )}

        <div className="input-group">
          <label>Add Web URL</label>
          <input 
            type="text" 
            placeholder="https://example.com" 
            value={url} 
            onChange={(e) => setUrl(e.target.value)}
          />
          <button 
            onClick={handleUrlSubmit} 
            disabled={ingestStatus === "processing"}
          >
            {ingestStatus === "processing" ? "⏳ Working..." : "🌐 Scrape Site"}
          </button>
        </div>

        <div className="input-group">
          <label>Upload PDF</label>
          <input 
            type="file" 
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
          <button 
            onClick={handleFileUpload}
            disabled={ingestStatus === "processing"}
          >
            {ingestStatus === "processing" ? "⏳ Working..." : "📄 Process PDF"}
          </button>
        </div>

        <button className="reset-btn" onClick={async () => {
            if(window.confirm("Are you sure you want to clear the brain?")) {
              await fetch("http://localhost:8000/reset", { method: "POST" });
              setMessages([]);
              setStatusMessage("✨ Brain Reset!");
            }
        }}>
          🗑️ Reset Brain
        </button>
      </div>

      {/* Chat Area */}
      <div className="chat-area">
        <div className="messages-list">
          {messages.length === 0 && (
            <div style={{textAlign: "center", color: "#94a3b8", marginTop: "100px"}}>
              <h3>👋 Welcome!</h3>
              <p>Add a URL or PDF to the left, then ask me anything.</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">
                {msg.text}
              </div>
              
              {msg.sources && msg.sources.length > 0 && (
                <details className="sources-dropdown">
                  <summary>📚 View {msg.sources.length} Verified Sources</summary>
                  <div className="sources-list">
                    {msg.sources.map((src, i) => (
                      <div key={i} className="source-item">
                        <small>{src.metadata.source}</small>
                        {src.text.substring(0, 150)}...
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          ))}
          
          {chatLoading && (
            <div className="message bot">
              <div className="message-content">
                <span className="loading-dots">Thinking</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <input 
            type="text" 
            placeholder="Ask a question about your documents..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            disabled={chatLoading}
          />
          <button onClick={sendMessage} disabled={chatLoading}>
            {chatLoading ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;