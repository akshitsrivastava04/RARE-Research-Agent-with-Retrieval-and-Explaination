import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './CouncilView.css';

// Component for a single Agent's panel
const AgentPanel = ({ agentName, role, response, index, isWinner }) => {
  return (
    <motion.div
      className={`agent-panel ${isWinner ? 'winner-panel' : ''}`}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.2, duration: 0.5 }}
    >
      <div className="agent-header">
        <h3 className="agent-name">{agentName}</h3>
        <span className="agent-role">{role}</span>
        {isWinner && <span className="winner-badge">🏆 Best Response</span>}
      </div>
      <div className="agent-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {response || "Thinking..."}
        </ReactMarkdown>
      </div>
    </motion.div>
  );
};

const CouncilView = ({ question, onBack, apiUrl, authHeaders }) => {
  const [status, setStatus] = useState('summoning'); // summoning, deliberating, voting, resolved
  const [individualResponses, setIndividualResponses] = useState([]);
  const [finalAnswer, setFinalAnswer] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchCouncil = async () => {
      try {
        setStatus('deliberating');

        // Call the backend
        const res = await fetch(`${apiUrl}/council_chat`, {
          method: 'POST',
          headers: authHeaders,
          body: JSON.stringify({ question })
        });

        if (!res.ok) throw new Error("Council unavailable");

        const data = await res.json();

        // Simulate a progressive reveal
        setIndividualResponses(data.individual_responses);

        // Wait a beat before showing the vote to add drama
        setTimeout(() => {
          setStatus('voting');
          setTimeout(() => {
            setFinalAnswer(data.final_answer);
            setStatus('resolved');
          }, 2000); // 2 seconds of "Judge Deliberating" animation
        }, 1000);

      } catch (err) {
        console.error(err);
        setError(err.message);
      }
    };

    fetchCouncil();
  }, [question, apiUrl, authHeaders]);

  // --- RENDER ---

  if (error) {
    return (
      <div className="council-container error-state">
        <h2>Council Error</h2>
        <p>{error}</p>
        <button onClick={onBack}>Return to Earth</button>
      </div>
    );
  }

  return (
    <div className="council-container">

      {/* HEADER */}
      <div className='council-top-bar'>
        <button onClick={onBack} className='back-button'>
          ← Back
        </button>
        <h2 className='council-title'>AI Council Session</h2>
      </div>

      <div className="council-question">
        <strong>Question:</strong> {question}
      </div>

      {/* MAIN STAGE */}
      <div className="council-stage">

        {/* SPLIT PANELS for AGENTS */}
        <div className="agents-grid">
          {/* We know the agents upfront, so we can render their slots immediately */}
          {["Cookie", "Captcha", "Plag"].map((name, idx) => {
            // Check if we have response data for this agent
            // The backend returns a list, order typically matches but let's be safe later.
            // For now, simple index matching or find by name if backend sends it.
            const agentData = individualResponses.find(a => a.agent_name === name) || individualResponses[idx];
            const hasResponse = !!agentData;

            return (
              <div key={name} className={`agent-panel ${!hasResponse ? 'loading-state' : ''}`}>
                <div className="agent-header">
                  <h3 className="agent-name">{name}</h3>
                  <span className="agent-role">
                    {name === "Cookie" ? "Summarizer" : name === "Captcha" ? "QA" : "Critic"}
                  </span>
                </div>

                <div className="agent-content">
                  {hasResponse ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {agentData.response}
                    </ReactMarkdown>
                  ) : (
                    <div className="loading-placeholder">
                      <div className="agent-header-skeleton"></div>
                      <div className="content-skeleton" style={{ width: '90%' }}></div>
                      <div className="content-skeleton" style={{ width: '80%' }}></div>
                      <div className="content-skeleton" style={{ width: '95%' }}></div>
                      <div className="spinner-small" style={{ margin: '2rem auto' }}></div>
                      <div style={{ textAlign: 'center', fontSize: '0.8rem', opacity: 0.6 }}>Thinking...</div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* JUDGE / FINAL ANSWER OVERLAY */}
        <AnimatePresence>
          {status === 'voting' && (
            <motion.div
              className="judge-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="judge-icon">⚖️</div>
              <h2>The Judge is voting...</h2>
            </motion.div>
          )}

          {status === 'resolved' && (
            <motion.div
              className="final-verdict-overlay"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 120 }}
            >
              <div className="verdict-card">
                <div className="verdict-header">
                  <span className="gavel-icon">⚖️</span>
                  <h3>Final Consensus</h3>
                  <button onClick={onBack} className="close-verdict">Close</button>
                </div>
                <div className="verdict-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {finalAnswer}
                  </ReactMarkdown>
                </div>
                <div className='verdict-actions'>
                  <button onClick={onBack} className='accept-button'>Accept Verdict & Return to Chat</button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
};

export default CouncilView;
