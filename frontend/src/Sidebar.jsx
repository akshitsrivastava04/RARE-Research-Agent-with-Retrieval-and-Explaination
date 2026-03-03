import React from 'react';
import './Sidebar.css';

const Sidebar = ({
    chats,
    activeChatId,
    onSelectChat,
    onNewChat,
    onDeleteChat,
    onLogout,
    userProfile
}) => {
    const username = userProfile?.username || "Guest";
    const initials = username.substring(0, 2).toUpperCase();

    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <span className="brand-icon"></span>
                <span>RARE</span>
            </div>

            <button className="new-research-btn" onClick={onNewChat}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                </svg>
                New Research
            </button>

            <div className="section-title">Today</div>
            <ul className="history-list">
                {chats.map(chat => (
                    <li
                        key={chat.id}
                        className={`history-item ${chat.id === activeChatId ? 'active' : ''}`}
                        onClick={() => onSelectChat(chat.id)}
                    >
                        <svg className="history-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z" />
                        </svg>
                        <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {chat.title || 'New Chat'}
                        </span>
                        <button
                            className="delete-chat-btn"
                            onClick={(e) => onDeleteChat(e, chat.id)}
                            title="Delete Chat"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </li>
                ))}
            </ul>

            <div className="user-profile" onClick={onLogout}>
                <div className="avatar-placeholder">{initials}</div>
                <div className="user-info">
                    <span className="user-name">{username}</span>
                    <span className="user-plan">Pro Plan</span>
                </div>
                <svg className="menu-dots" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M3 9.5a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3z" />
                </svg>
            </div>
        </div>
    );
};

export default Sidebar;
