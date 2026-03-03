import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './MainContent.css';

const MainContent = ({
    activeChat,
    input,
    setInput,
    onSend,
    isLoading,
    attachedFiles,
    onAttachClick,
    onFileChange,
    onRemoveFile,
    fileInputRef,
    isCouncilMode,
    setIsCouncilMode,
    webSearchEnabled,
    setWebSearchEnabled
}) => {
    const scrollRef = useRef(null);

    // Auto-scroll to bottom of chat
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [activeChat?.messages]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSend();
        }
    };

    const handleSuggestionClick = (text) => {
        setInput(text);
    };

    const suggestions = [
        "Explain quantum entanglement with recent papers",
        "Compare transformer architectures in NLP",
        "Summarize recent advances in CRISPR",
        "What are the key findings on dark matter in 2024?"
    ];

    const isLanding = !activeChat || !activeChat.messages || activeChat.messages.length <= 1;

    return (
        <div className="main-content-area">
            <div className="main-content-bg"></div>

            <div className="scroll-area" ref={scrollRef}>
                {isLanding ? (
                    <div className="landing-container">
                        <div className="hero-logo-section">
                            <div className="hero-logo">
                                <span className="hero-diamond"></span>
                                RARE
                            </div>
                            <div className="hero-subtitle">
                                Research Agent with Retrieval & Explanation
                            </div>
                        </div>

                        <div className="mode-toggle">
                            <button
                                className={`mode-btn ${!isCouncilMode ? 'active' : ''}`}
                                onClick={() => setIsCouncilMode(false)}
                            >
                                Quick Answer
                            </button>
                            <button
                                className={`mode-btn ${isCouncilMode ? 'active' : ''}`}
                                onClick={() => setIsCouncilMode(true)}
                            >
                                Deep Dive
                            </button>
                            <button className="mode-btn">
                                Literature Review
                            </button>
                        </div>

                        <div className="suggestions-list">
                            {suggestions.map((text, idx) => (
                                <div
                                    key={idx}
                                    className="suggestion-item"
                                    onClick={() => handleSuggestionClick(text)}
                                >
                                    <span className="suggestion-number">[{idx + 1}]</span>
                                    <span>{text}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="chat-messages-container">
                        {activeChat.messages.map((msg, idx) => (
                            <div key={idx} className={`chat-bubble ${msg.role}`}>
                                {msg.files && msg.files.length > 0 && (
                                    <div className="input-files">
                                        {msg.files.map((f, i) => (
                                            <span key={i} className="input-file-pill">
                                                {f.name}
                                            </span>
                                        ))}
                                    </div>
                                )}

                                {msg.role === 'bot' ? (
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                                ) : (
                                    msg.text
                                )}
                            </div>
                        ))}
                        {isLoading && (
                            <div className="chat-bubble bot">
                                <span className="typing-dots">Thinking...</span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="input-position-wrapper">
                <div className="input-container">
                    {attachedFiles.length > 0 && (
                        <div className="input-files">
                            {attachedFiles.map(file => (
                                <div key={file.name} className="input-file-pill">
                                    <span>{file.name}</span>
                                    <button
                                        onClick={() => onRemoveFile(file.name)}
                                        style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', marginLeft: '5px' }}
                                    >
                                        ×
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="input-top-row">
                        {isCouncilMode && <span className="deep-dive-tag">Deep Dive</span>}
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask a research question..."
                            className="input-field"
                            disabled={isLoading}
                        />
                    </div>

                    <div className="input-footer">
                        <div style={{ display: 'flex', gap: '15px' }}>
                            <button
                                onClick={onAttachClick}
                                className="footer-icon"
                                style={{ background: 'none', border: 'none', padding: 0 }}
                                title="Attach file"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" style={{ width: '20px', height: '20px' }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94a3 3 0 1 1 4.243 4.242l-9.88 9.88a1.5 1.5 0 0 1-2.12-2.121l9.88-9.88" />
                                </svg>
                            </button>
                            <button
                                className={`footer-icon ${webSearchEnabled ? 'active' : ''}`}
                                onClick={() => setWebSearchEnabled(!webSearchEnabled)}
                                style={{
                                    background: 'none',
                                    border: 'none',
                                    padding: 0,
                                    color: webSearchEnabled ? '#3b82f6' : 'inherit'
                                }}
                                title="Web Search"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" style={{ width: '20px', height: '20px' }}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 0 0 8.716-6.747M12 21a9.004 9.004 0 0 1-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 0 1 7.843 4.582M12 3a8.997 8.997 0 0 0-7.843 4.582m15.686 0A11.953 11.953 0 0 1 12 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0 1 21.812 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0 1 12 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 0 1 3 12c0-1.605.42-3.113 1.157-4.418" />
                                </svg>
                            </button>
                        </div>

                        <button
                            className="send-btn-rare"
                            onClick={onSend}
                            disabled={isLoading || (!input.trim() && attachedFiles.length === 0)}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{ width: '16px', height: '16px' }}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L6 12Zm0 0h7.5" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>

            <input
                type="file"
                ref={fileInputRef}
                onChange={onFileChange}
                style={{ display: 'none' }}
                accept="text/plain, .pdf, .md"
                multiple
            />
        </div>
    );
};

export default MainContent;
