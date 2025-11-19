"""
Transformer-Based Dialogue Manager with Multi-Turn Context
Handles conversational flow and context management
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages multi-turn conversation context"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.metadata: Dict = {
            'session_id': None,
            'start_time': datetime.now(),
            'user_info': {},
            'escalation_count': 0,
            'sentiment_history': []
        }

    def add_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add a conversation turn to history"""
        turn = {
            'role': role,
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversation_history.append(turn)

        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def get_history(self, n_turns: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        if n_turns:
            return self.conversation_history[-n_turns * 2:]
        return self.conversation_history

    def get_context_string(self, n_turns: int = 5) -> str:
        """Get formatted context string for model input"""
        recent_turns = self.get_history(n_turns)
        context_parts = []

        for turn in recent_turns:
            prefix = "User:" if turn['role'] == 'user' else "Assistant:"
            context_parts.append(f"{prefix} {turn['text']}")

        return "\n".join(context_parts)

    def clear(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.metadata['start_time'] = datetime.now()


class DialogueManager:
    """Main dialogue manager with transformer-based response generation"""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = None,
        max_context_turns: int = 5
    ):
        """
        Initialize dialogue manager

        Args:
            model_name: HuggingFace model name for dialogue
            device: Device to run model on (cuda/cpu)
            max_context_turns: Maximum conversation turns to keep in context
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_turns = max_context_turns

        logger.info(f"Loading dialogue model: {model_name} on {self.device}")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Initialize text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            logger.info("Dialogue model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading dialogue model: {e}")
            raise

        # Conversation contexts for multiple sessions
        self.active_contexts: Dict[str, ConversationContext] = {}

    def create_session(self, session_id: str) -> ConversationContext:
        """Create a new conversation session"""
        context = ConversationContext(max_history=self.max_context_turns)
        context.metadata['session_id'] = session_id
        self.active_contexts[session_id] = context

        logger.info(f"Created new session: {session_id}")
        return context

    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create new one"""
        if session_id not in self.active_contexts:
            return self.create_session(session_id)
        return self.active_contexts[session_id]

    def generate_response(
        self,
        user_input: str,
        session_id: str,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_context: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate response using transformer model

        Args:
            user_input: User's input text
            session_id: Session identifier
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_context: Whether to use conversation context

        Returns:
            Tuple of (response_text, metadata)
        """
        context = self.get_or_create_session(session_id)

        # Add user input to context
        context.add_turn('user', user_input)

        try:
            # Build input with context
            if use_context:
                context_str = context.get_context_string(self.max_context_turns)
                input_text = f"{context_str}\nAssistant:"
            else:
                input_text = f"User: {user_input}\nAssistant:"

            # Encode input
            input_ids = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Decode response
            full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract only the new response
            response = full_response.split("Assistant:")[-1].strip()

            # Clean up response
            response = self._clean_response(response)

            # Add response to context
            metadata = {
                'model': 'transformer',
                'temperature': temperature,
                'context_used': use_context
            }
            context.add_turn('assistant', response, metadata)

            logger.info(f"Generated response for session {session_id}")

            return response, metadata

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = "I apologize, but I'm having trouble processing your request. Could you please rephrase?"
            context.add_turn('assistant', error_response, {'error': str(e)})
            return error_response, {'error': str(e)}

    def _clean_response(self, response: str) -> str:
        """Clean and format response text"""
        # Remove any user prompts that might have leaked
        if "User:" in response:
            response = response.split("User:")[0]

        # Remove excessive whitespace
        response = " ".join(response.split())

        # Truncate at sentence boundary if too long
        if len(response) > 300:
            sentences = response.split('. ')
            response = '. '.join(sentences[:2]) + '.'

        return response.strip()

    def end_session(self, session_id: str) -> Dict:
        """End a conversation session and return summary"""
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            summary = {
                'session_id': session_id,
                'duration': (datetime.now() - context.metadata['start_time']).seconds,
                'turn_count': len(context.conversation_history) // 2,
                'escalated': context.metadata['escalation_count'] > 0
            }

            # Remove from active contexts
            del self.active_contexts[session_id]

            logger.info(f"Ended session {session_id}: {summary}")
            return summary

        return {}

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id in self.active_contexts:
            return self.active_contexts[session_id].get_history()
        return []

    def export_session(self, session_id: str, filepath: str):
        """Export session conversation to file"""
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            data = {
                'session_id': session_id,
                'metadata': context.metadata,
                'conversation': context.get_history()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported session {session_id} to {filepath}")


# Convenience function for quick testing
def test_dialogue_manager():
    """Test the dialogue manager"""
    dm = DialogueManager()
    session_id = "test_session_001"

    print("Dialogue Manager Test")
    print("-" * 50)

    test_inputs = [
        "Hello, I need help with my account",
        "I can't log in to my account",
        "I've tried resetting my password but didn't receive the email",
        "What should I do now?"
    ]

    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        response, metadata = dm.generate_response(user_input, session_id)
        print(f"Bot: {response}")

    # End session
    summary = dm.end_session(session_id)
    print(f"\nSession Summary: {summary}")


if __name__ == "__main__":
    test_dialogue_manager()
