import asyncio
from collections import Counter
from dotenv import load_dotenv
from services.groq_api import GroqAPI  # Use the Groq client

load_dotenv()


class Agent:
    """
    Represents an AI agent using a specific model via the Groq API.
    """
    def __init__(self, name: str, role: str, model_id: str):
        self.name = name
        self.role = role
        self.model_id = model_id  # Groq model ID
        
        try:
            self.client = GroqAPI(model_id=self.model_id)
        except (ValueError, EnvironmentError) as e:
            print(f"[AGENT INIT ERROR] {self.name}: {e}")
            self.client = None 

    async def ask(self, question: str, context: str = "") -> str:
        """
        Ask the agent a question, optionally with context from other agents.
        Returns the text response or an error string.
        """
        if not self.client:
            return f"Groq API Error: Agent '{self.name}' failed to initialize."
            
        prompt = (
            f"You are a helpful AI assistant. Your name is {self.name} and your role is {self.role}.\n"
            f"--- QUESTION ---\n"
            f"{question}\n"
        )
        
        if context:
            prompt += f"\n--- OTHER AGENTS' RESPONSES ---\n{context}"

        response = await self.client.query(prompt)
        return response.strip() if isinstance(response, str) else str(response)

    async def run_council(self, question: str, other_agents: list) -> dict:
        """
        Executes the Council Flow:
        1. All agents (including self) answer the question in parallel.
        2. A 'Judge' (self, acting as chair) reviews all answers and picks/synthesizes the best one.
        Returns:
        {
            "individual_responses": [{"agent_name": str, "response": str}, ...],
            "answer": str, # The final verdict
            "tokens_used": int
        }
        """
        all_agents = [self] + other_agents
        
        # --- STEP 1: PARALLEL RESPONSES (The Council Speaks) ---
        print(f"\n--- COUNCIL SESSION: '{question}' ---")
        tasks = [agent.ask(question) for agent in all_agents]
        responses = await asyncio.gather(*tasks)
        
        individual_responses = []
        for agent, resp in zip(all_agents, responses):
            individual_responses.append({
                "agent_name": agent.name, 
                "role": agent.role,
                "response": resp
            })
        
        # --- STEP 2: THE JUDGE DECIDES ---
        # Construct the context for the Judge
        candidates_text = ""
        for i, item in enumerate(individual_responses):
            candidates_text += f"--- Candidate {i+1} ({item['agent_name']} - {item['role']}) ---\n{item['response']}\n\n"
            
        judge_prompt = (
            f"You are the Head of the AI Council. You have received the following responses to the question: '{question}'.\n\n"
            f"{candidates_text}\n"
            f"Your task is to provide the FINAL, BEST ANSWER.\n"
            f"1. You can pick the best response verbatim if it is perfect.\n"
            f"2. Or you can synthesize a new answer combining the best parts of the candidates.\n"
            f"3. Explain briefly why you chose this result if you want, but ensure the main answer is clear.\n"
            f"Format your response in clean Markdown."
        )
        
        print("\n--- JUDGE DELIBERATING ---")
        final_verdict = await self.client.query(judge_prompt)
        
        # --- Calculate Token Usage ---
        tokens_this_call = 0
        all_agents_plus_judge = all_agents + [self] # Self is used twice (worker + judge)
        for agent in all_agents_plus_judge:
            if agent.client and agent.client.last_response_metadata:
                token_usage = agent.client.last_response_metadata.get('token_usage', {})
                tokens_this_call += token_usage.get('total_tokens', 0)
        
        print(f"\n--- COUNCIL VERDICT ---\n{final_verdict[:100]}...")
        
        return {
            "individual_responses": individual_responses,
            "answer": final_verdict,
            "tokens_used": tokens_this_call
        }
    
    # Deprecated but kept for compatibility if needed
    async def discuss_with(self, question: str, other_agents: list, rounds: int = 2) -> dict:
        return await self.run_council(question, other_agents)

    def get_consensus(self, responses: list) -> str:
        # Legacy method
        return responses[0] if responses else ""


# --- Example usage ---
if __name__ == "__main__":
    async def main():
        
        print("--- Initializing Agents ---")
        # Initialize agents with Groq model IDs
        agent1 = Agent("Cookie", "Summarizer", model_id="llama-3.1-8b-instant")
        agent2 = Agent("Captcha", "QA", model_id="moonshotai/kimi-k2-instruct")
        agent3 = Agent("Plag", "Critic", model_id="qwen/qwen3-32b")
        
        agents = [agent1, agent2, agent3]
        
        # Check if any agent failed to initialize
        if any(agent.client is None for agent in agents):
            print("One or more agents failed to initialize. Check GROQ_API_KEY. Exiting.")
            return

        question = "Explain the concept of Retrieval-Augmented Generation (RAG) in AI."
        print(f"\nAsking the agents to discuss: '{question}'")
        
        # Start discussion with agent1
        consensus_result = await agent1.discuss_with(question, [agent2, agent3])
        
        print("\n--- FINAL CONSENSUS RESULT ---")
        print(consensus_result['answer'])

        # Show total *daily* tokens used across all models in this session
        total_tokens = sum(agent.client.model_daily_tokens for agent in agents)
        print(f"\n--- TOTAL DAILY USAGE (ALL MODELS) ---")
        print(f"Total daily tokens tracked across all models: {total_tokens}")
    asyncio.run(main())