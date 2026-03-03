import React, { useState, useEffect, useRef } from "react";
import "./App.css";

// --- COMPONENTS ---
import LoginPage from './LoginPage';
import Sidebar from './Sidebar';
import MainContent from './MainContent';
import CouncilView from './CouncilView';

// --- PDF.js IMPORTS ---
import * as pdfjsLib from 'pdfjs-dist';
pdfjsLib.GlobalWorkerOptions.workerSrc = "//cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.mjs";

// Define the backend API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const AVAILABLE_MODELS = {
  "moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct-0905",
  "llama-3.1-8b-instant": "llama-3.1-8b-instant",
  'openai/gpt-oss-20b': "openai/gpt-oss-20b"
};

// --- Auth Header Helper ---
const getAuthHeaders = () => {
  const token = localStorage.getItem('my_app_token');
  if (!token) {
    return { 'Content-Type': 'application/json' };
  }
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };
};

// Main App Component
function App() {
  // --- Auth State ---
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // --- Chat State ---
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [userProfile, setUserProfile] = useState({ username: "Guest" });

  const [input, setInput] = useState("");
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelUsage, setModelUsage] = useState({});
  const [lastCallTokens, setLastCallTokens] = useState(0);
  const [selectedModel, setSelectedModel] = useState("llama-3.1-8b-instant");
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);

  // --- Council Mode State ---
  const [councilQuestion, setCouncilQuestion] = useState(null);
  const [isCouncilMode, setIsCouncilMode] = useState(false);

  const activeChat = chats.find(chat => chat.id === activeChatId);

  const fileInputRef = useRef(null);
  const chatsRef = useRef(chats);

  useEffect(() => {
    chatsRef.current = chats;
  }, [chats]);

  // --- Auth Check Effect ---
  useEffect(() => {
    const token = localStorage.getItem('my_app_token');
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  // --- Chat Loading Effect ---
  useEffect(() => {
    if (isAuthenticated) {
      fetchUserChats();
      fetchUserProfile();
    }
  }, [isAuthenticated]);

  const fetchUserProfile = async () => {
    try {
      const response = await fetch(`${API_URL}/me`, { headers: getAuthHeaders() });
      if (response.ok) {
        const data = await response.json();
        setUserProfile(data);
      }
    } catch (error) {
      console.error("Error fetching profile:", error);
    }
  };

  // --- URL CLEANUP EFFECT ---
  useEffect(() => {
    return () => {
      chatsRef.current.forEach(chat => {
        chat.messages.forEach(msg => {
          if (msg.files && Array.isArray(msg.files)) {
            msg.files.forEach(file => {
              if (file.url && file.url.startsWith('blob:')) {
                URL.revokeObjectURL(file.url);
              }
            });
          }
        });
      });
    };
  }, []);

  // --- Fetch User Chats Function ---
  const fetchUserChats = async () => {
    try {
      const headers = getAuthHeaders();
      const response = await fetch(`${API_URL}/sessions`, { headers });
      if (!response.ok) {
        if (response.status === 401) {
          handleLogout();
        }
        throw new Error('Failed to fetch chats');
      }
      const data = await response.json();

      const formattedChats = data.map(chat => ({
        id: chat._id,
        backendSessionId: chat._id,
        title: chat.title,
        messages: chat.messages
      }));

      setChats(formattedChats);
      if (formattedChats.length > 0) {
        setActiveChatId(formattedChats[0].id);
      } else {
        handleNewChat();
      }
    } catch (error) {
      console.error("Error fetching user chats:", error);
      handleNewChat();
    }
  };

  // --- HELPER FUNCTIONS ---
  const readFileAsText = (file) => {
    if (file.type === 'application/pdf') {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = async () => {
          try {
            const pdf = await pdfjsLib.getDocument(reader.result).promise;
            let fullText = '';
            for (let i = 1; i <= pdf.numPages; i++) {
              const page = await pdf.getPage(i);
              const textContent = await page.getTextContent();
              fullText += textContent.items.map(item => item.str).join(' ') + '\n';
            }
            resolve(fullText);
          } catch (error) {
            console.error("Error parsing PDF:", error);
            reject(new Error("Could not parse PDF file."));
          }
        };
        reader.onerror = (error) => reject(error);
      });
    } else {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsText(file, 'UTF-8');
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
      });
    }
  };

  const addBotMessageToChat = (text) => {
    const botMessage = { role: "bot", text: text.trim() };
    setChats(prevChats => prevChats.map(chat =>
      chat.id === activeChatId ? { ...chat, messages: [...(chat.messages || []), botMessage] } : chat
    ));
    // Force a small state update to ensure UI re-renders if needed
    setIsLoading(prev => prev);
  };

  // --- CHAT FUNCTIONS ---
  const handleSend = async () => {
    const userText = input.trim();
    const filesToUpload = attachedFiles;
    if (isLoading || (!userText && filesToUpload.length === 0)) return;

    setIsLoading(true);

    const authHeaders = getAuthHeaders();

    let currentChat = chats.find(c => c.id === activeChatId);
    let activeSessionId = currentChat ? currentChat.backendSessionId : null;

    // 1. Optimistic UI update
    const userMessage = { role: "user", text: userText };
    if (filesToUpload.length > 0) {
      userMessage.files = filesToUpload.map(file => ({
        name: file.name, size: file.size, type: file.type, url: URL.createObjectURL(file)
      }));
    }

    let updatedChat = { ...currentChat };
    updatedChat.messages = [...(updatedChat.messages || []), userMessage];

    // Title update logic
    const firstUserMessage = updatedChat.messages.filter(m => m.role === 'user').length === 1;
    if (firstUserMessage && !activeSessionId) {
      const firstUserMessageText = userText || (filesToUpload.length > 0 ? filesToUpload[0].name : 'New Chat');
      updatedChat.title = firstUserMessageText.substring(0, 40) + (firstUserMessageText.length > 40 ? '...' : '');
    }

    setChats(chats.map(chat => chat.id === activeChatId ? updatedChat : chat));
    setInput("");
    setAttachedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = null;

    try {
      // Step 2: Handle file uploads
      if (filesToUpload.length > 0) {
        const processingMsg = { role: "bot", text: "Processing your document(s)..." };
        setChats(prev => prev.map(c => c.id === activeChatId ? { ...c, messages: [...c.messages, processingMsg] } : c));

        for (const file of filesToUpload) {
          let textForUpload;
          try { textForUpload = await readFileAsText(file); }
          catch (e) {
            console.error("File read error:", e);
            addBotMessageToChat(`Sorry, I had trouble reading ${file.name}: ${e.message}. Skipping this file.`);
            continue;
          }

          const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            headers: authHeaders,
            body: JSON.stringify({ text: textForUpload, session_id: activeSessionId })
          });

          if (!response.ok) {
            const err = await response.json();
            throw new Error(`Upload failed for ${file.name}: ${err.detail || response.statusText}`);
          }

          const data = await response.json();
          activeSessionId = data.session_id;

          setChats(prevChats => prevChats.map(chat =>
            chat.id === activeChatId
              ? { ...chat, backendSessionId: activeSessionId, id: activeSessionId }
              : chat
          ));
          setActiveChatId(activeSessionId);
        }

        if (!userText) {
          addBotMessageToChat("Your document(s) have been processed. What would you like to ask about them?");
          setIsLoading(false);
          await fetchUserChats(); // Refresh session list to show new session if any
          return;
        }
      }

      // Step 3: Call /chat
      if (userText) {
        const chatResponse = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: authHeaders,
          body: JSON.stringify({
            question: userText,
            session_id: activeSessionId,
            model_id: selectedModel,
            web_search: webSearchEnabled
          })
        });

        if (!chatResponse.ok) {
          const err = await chatResponse.json();
          throw new Error(`Chat failed: ${err.detail || chatResponse.statusText}`);
        }

        const chatData = await chatResponse.json();
        const newSessionId = chatData.session_id;
        const botMessage = { role: "bot", text: chatData.answer.trim() };
        let newActiveId = activeChatId;

        setChats(prevChats => {
          const updated = prevChats.map(chat => {
            if (chat.id === activeChatId) {
              const isNewChat = !chat.backendSessionId;
              if (isNewChat) {
                newActiveId = newSessionId;
              }
              // If we added a "Processing" message, we might want to replace it or just append. 
              // User said "does not update message from processing to generated", 
              // but we are using a global "Thinking..." indicator and appending bot messages.
              // Let's ensure the state update is clean.
              return {
                ...chat,
                messages: [...(chat.messages || []), botMessage],
                id: isNewChat ? newSessionId : chat.id,
                backendSessionId: isNewChat ? newSessionId : chat.backendSessionId
              };
            }
            return chat;
          });
          return updated;
        });

        if (activeChatId !== newActiveId) {
          setActiveChatId(newActiveId);
        }

        // Usage data handling
        if (chatData.usage_data && chatData.usage_data.models) {
          const newModelTotals = {};
          for (const [modelId, usageData] of Object.entries(chatData.usage_data.models)) {
            let tokens = NaN;
            if (usageData && typeof usageData === 'object' && usageData.total_tokens_used !== undefined) {
              tokens = Number(usageData.total_tokens_used);
            } else if (usageData !== undefined && typeof usageData !== 'object') {
              tokens = Number(usageData);
            }
            if (!isNaN(tokens)) {
              newModelTotals[modelId] = tokens;
            }
          }
          setModelUsage(prevUsage => ({ ...prevUsage, ...newModelTotals }));
        }
        setLastCallTokens(chatData.tokens_from_this_call || 0);

        // Refresh session list to ensure new sessions appear in sidebar
        await fetchUserChats();
      }

    } catch (error) {
      console.error("API Error:", error);
      addBotMessageToChat(`Sorry, something went wrong: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    const newId = `chat-${Date.now()}`;
    const newChat = {
      id: newId,
      title: 'New Chat',
      messages: [{ role: "bot", text: "Hello! Please upload a document or ask a question." }],
      backendSessionId: null
    };

    if (chats.length > 0 && chats[0].backendSessionId === null) {
      setActiveChatId(chats[0].id);
      return;
    }

    setChats([newChat, ...chats]);
    setActiveChatId(newId);
  };

  const handleDeleteChat = async (e, chatId) => {
    if (e) e.stopPropagation();

    const chatToDelete = chatsRef.current.find(chat => chat.id === chatId);
    if (!chatToDelete) return;

    if (chatToDelete.backendSessionId) {
      try {
        const response = await fetch(`${API_URL}/sessions/${chatToDelete.backendSessionId}`, {
          method: 'DELETE',
          headers: getAuthHeaders()
        });
        if (!response.ok) {
          console.error("Failed to delete chat on backend");
        }
      } catch (error) {
        console.error("Error deleting chat:", error);
      }
    }

    if (chatToDelete.messages) {
      chatToDelete.messages.forEach(msg => {
        if (msg.files && Array.isArray(msg.files)) {
          msg.files.forEach(file => {
            if (file.url && file.url.startsWith('blob:')) {
              URL.revokeObjectURL(file.url);
            }
          });
        }
      });
    }

    let newChats = chats.filter(chat => chat.id !== chatId);

    if (newChats.length === 0) {
      const newId = `chat-${Date.now()}`;
      newChats = [{
        id: newId,
        title: 'New Chat',
        messages: [{ role: "bot", text: "Hello! Please upload a document or ask a question." }],
        backendSessionId: null
      }];
    }

    setChats(newChats);

    if (activeChatId === chatId) {
      setActiveChatId(newChats[0].id);
    }
  };

  // --- FILE HANDLERS ---
  const handleAttachClick = () => {
    if (isLoading) return;
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setAttachedFiles(Array.from(e.target.files));
    }
  };

  const removeFileFromList = (fileName) => {
    setAttachedFiles(prevFiles => prevFiles.filter(file => file.name !== fileName));
    if (fileInputRef.current) {
      fileInputRef.current.value = null;
    }
  };

  // --- Auth Handlers ---
  const handleLoginSuccess = (token) => {
    localStorage.setItem('my_app_token', token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('my_app_token');
    setIsAuthenticated(false);
    setChats([]);
    setActiveChatId(null);
  };


  // --- Council Handlers ---
  const handleSummonCouncil = () => {
    if (!input.trim()) return;
    setCouncilQuestion(input.trim());
    setInput("");
  };

  const handleCloseCouncil = () => {
    setCouncilQuestion(null);
    if (isAuthenticated) fetchUserChats();
  };

  const handleGeneralSend = () => {
    if (isCouncilMode) {
      handleSummonCouncil();
    } else {
      handleSend();
    }
  };

  if (!isAuthenticated) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} apiUrl={API_URL} />;
  }

  // --- CONDITIONAL RENDER: Council View OR Main Layout ---
  if (councilQuestion) {
    return (
      <CouncilView
        question={councilQuestion}
        onBack={handleCloseCouncil}
        apiUrl={API_URL}
        authHeaders={getAuthHeaders()}
      />
    );
  }

  return (
    <div className="app-container">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={setActiveChatId}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onLogout={handleLogout}
        userProfile={userProfile}
      />
      <MainContent
        activeChat={activeChat}
        input={input}
        setInput={setInput}
        onSend={handleGeneralSend}
        isLoading={isLoading}
        attachedFiles={attachedFiles}
        onAttachClick={handleAttachClick}
        onFileChange={handleFileChange}
        onRemoveFile={removeFileFromList}
        fileInputRef={fileInputRef}
        isCouncilMode={isCouncilMode}
        setIsCouncilMode={setIsCouncilMode}
        webSearchEnabled={webSearchEnabled}
        setWebSearchEnabled={setWebSearchEnabled}
      />
    </div>
  );
}

export default App;
