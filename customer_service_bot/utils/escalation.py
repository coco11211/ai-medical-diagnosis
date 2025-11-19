"""
Escalation Logic and Routing System
Determines when to escalate to human agents
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EscalationManager:
    """
    Manages escalation logic and routing to human agents
    """

    def __init__(
        self,
        sentiment_threshold: float = -0.5,
        urgency_threshold: float = 7.0,
        unresolved_turns_threshold: int = 3,
        enable_auto_escalation: bool = True
    ):
        """
        Initialize escalation manager

        Args:
            sentiment_threshold: Negative sentiment threshold for escalation
            urgency_threshold: Urgency score threshold for escalation
            unresolved_turns_threshold: Number of unresolved turns before escalation
            enable_auto_escalation: Whether to enable automatic escalation
        """
        self.sentiment_threshold = sentiment_threshold
        self.urgency_threshold = urgency_threshold
        self.unresolved_turns_threshold = unresolved_turns_threshold
        self.enable_auto_escalation = enable_auto_escalation

        # Track escalation events
        self.escalation_history: List[Dict] = []

        # Escalation rules
        self.escalation_rules = self._initialize_rules()

        logger.info("Escalation manager initialized")

    def _initialize_rules(self) -> List[Dict]:
        """Initialize escalation rules"""
        return [
            {
                'id': 'high_urgency',
                'description': 'High urgency detected',
                'check': lambda ctx: ctx.get('urgency', {}).get('score', 0) >= self.urgency_threshold,
                'priority': 'high',
                'reason': 'High urgency keywords detected'
            },
            {
                'id': 'very_negative_sentiment',
                'description': 'Very negative sentiment',
                'check': lambda ctx: ctx.get('sentiment', {}).get('score', 0) <= self.sentiment_threshold,
                'priority': 'high',
                'reason': 'Customer is very upset or frustrated'
            },
            {
                'id': 'complaint_intent',
                'description': 'Complaint detected',
                'check': lambda ctx: ctx.get('intent', '') == 'complaint',
                'priority': 'high',
                'reason': 'Formal complaint filed'
            },
            {
                'id': 'unresolved_multiple_turns',
                'description': 'Multiple unresolved turns',
                'check': lambda ctx: ctx.get('unresolved_turns', 0) >= self.unresolved_turns_threshold,
                'priority': 'medium',
                'reason': 'Unable to resolve issue after multiple attempts'
            },
            {
                'id': 'low_confidence',
                'description': 'Low AI confidence',
                'check': lambda ctx: ctx.get('confidence', 1.0) < 0.4,
                'priority': 'medium',
                'reason': 'AI is uncertain about how to help'
            },
            {
                'id': 'explicit_request',
                'description': 'Explicit request for human',
                'check': lambda ctx: self._check_human_request(ctx.get('text', '')),
                'priority': 'high',
                'reason': 'Customer explicitly requested human agent'
            },
            {
                'id': 'sentiment_declining',
                'description': 'Declining sentiment trend',
                'check': lambda ctx: ctx.get('sentiment_trend', '') == 'declining' and
                                     ctx.get('sentiment', {}).get('score', 0) < -0.3,
                'priority': 'medium',
                'reason': 'Customer satisfaction is declining'
            },
            {
                'id': 'complex_issue',
                'description': 'Complex multi-faceted issue',
                'check': lambda ctx: len(ctx.get('entities', [])) > 5 and
                                     len(ctx.get('text', '').split()) > 50,
                'priority': 'low',
                'reason': 'Issue appears complex and multi-faceted'
            }
        ]

    def _check_human_request(self, text: str) -> bool:
        """Check if user is requesting a human agent"""
        human_keywords = [
            'speak to a person',
            'talk to a human',
            'human agent',
            'real person',
            'speak to agent',
            'talk to representative',
            'speak to manager',
            'human support',
            'live agent',
            'customer service representative'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in human_keywords)

    def should_escalate(self, context: Dict) -> Tuple[bool, Dict]:
        """
        Determine if conversation should be escalated

        Args:
            context: Context dictionary with sentiment, intent, urgency, etc.

        Returns:
            Tuple of (should_escalate, escalation_info)
        """
        if not self.enable_auto_escalation:
            # Check only explicit requests
            if self._check_human_request(context.get('text', '')):
                return True, {
                    'reason': 'Customer explicitly requested human agent',
                    'priority': 'high',
                    'triggered_rules': ['explicit_request']
                }
            return False, {}

        # Check all escalation rules
        triggered_rules = []
        highest_priority = 'low'

        for rule in self.escalation_rules:
            try:
                if rule['check'](context):
                    triggered_rules.append(rule['id'])

                    # Update highest priority
                    if rule['priority'] == 'high':
                        highest_priority = 'high'
                    elif rule['priority'] == 'medium' and highest_priority == 'low':
                        highest_priority = 'medium'

            except Exception as e:
                logger.warning(f"Error checking rule {rule['id']}: {e}")

        # Escalate if any high priority rules triggered
        # or if multiple medium/low priority rules triggered
        should_escalate = False
        reasons = []

        if triggered_rules:
            if highest_priority == 'high':
                should_escalate = True
            elif len(triggered_rules) >= 2:  # Multiple rules triggered
                should_escalate = True

            # Collect reasons
            for rule_id in triggered_rules:
                rule = next(r for r in self.escalation_rules if r['id'] == rule_id)
                reasons.append(rule['reason'])

        escalation_info = {
            'should_escalate': should_escalate,
            'priority': highest_priority,
            'triggered_rules': triggered_rules,
            'reasons': reasons,
            'context_summary': self._summarize_context(context)
        }

        if should_escalate:
            self._log_escalation(escalation_info)
            logger.info(f"Escalation triggered: {', '.join(triggered_rules)}")

        return should_escalate, escalation_info

    def _summarize_context(self, context: Dict) -> Dict:
        """Summarize context for escalation"""
        return {
            'sentiment': context.get('sentiment', {}).get('label', 'unknown'),
            'sentiment_score': context.get('sentiment', {}).get('score', 0),
            'intent': context.get('intent', 'unknown'),
            'urgency_level': context.get('urgency', {}).get('level', 'unknown'),
            'urgency_score': context.get('urgency', {}).get('score', 0),
            'unresolved_turns': context.get('unresolved_turns', 0),
            'conversation_duration': context.get('duration_minutes', 0)
        }

    def _log_escalation(self, escalation_info: Dict):
        """Log escalation event"""
        self.escalation_history.append({
            'timestamp': datetime.now().isoformat(),
            'priority': escalation_info['priority'],
            'triggered_rules': escalation_info['triggered_rules'],
            'reasons': escalation_info['reasons']
        })

    def get_routing_info(self, escalation_info: Dict, session_context: Optional[Dict] = None) -> Dict:
        """
        Get routing information for escalated conversation

        Args:
            escalation_info: Escalation information from should_escalate()
            session_context: Optional session context

        Returns:
            Routing information
        """
        # Determine department based on intent
        intent = session_context.get('intent', 'general_inquiry') if session_context else 'general_inquiry'

        department_mapping = {
            'billing_payment': 'Billing Department',
            'technical_support': 'Technical Support',
            'account_issue': 'Account Services',
            'complaint': 'Customer Relations',
            'return_exchange': 'Returns Department',
            'order_status': 'Order Fulfillment',
        }

        department = department_mapping.get(intent, 'General Support')

        # Prepare handoff information
        routing_info = {
            'department': department,
            'priority': escalation_info.get('priority', 'medium'),
            'queue': self._get_queue_name(escalation_info['priority']),
            'estimated_wait_time': self._estimate_wait_time(escalation_info['priority']),
            'agent_requirements': self._get_agent_requirements(intent),
            'context_summary': escalation_info.get('context_summary', {}),
            'conversation_summary': self._create_conversation_summary(session_context)
        }

        return routing_info

    def _get_queue_name(self, priority: str) -> str:
        """Get queue name based on priority"""
        queue_mapping = {
            'high': 'Priority Queue',
            'medium': 'Standard Queue',
            'low': 'General Queue'
        }
        return queue_mapping.get(priority, 'General Queue')

    def _estimate_wait_time(self, priority: str) -> str:
        """Estimate wait time based on priority"""
        # This would ideally integrate with actual queue management
        wait_times = {
            'high': '2-5 minutes',
            'medium': '5-10 minutes',
            'low': '10-15 minutes'
        }
        return wait_times.get(priority, '10-15 minutes')

    def _get_agent_requirements(self, intent: str) -> List[str]:
        """Get required agent skills for intent"""
        requirements_mapping = {
            'billing_payment': ['billing', 'payments', 'refunds'],
            'technical_support': ['technical', 'troubleshooting'],
            'account_issue': ['account_management'],
            'complaint': ['escalation_handling', 'customer_relations'],
            'return_exchange': ['returns', 'exchanges'],
        }
        return requirements_mapping.get(intent, ['general_support'])

    def _create_conversation_summary(self, session_context: Optional[Dict]) -> str:
        """Create human-readable conversation summary for agent"""
        if not session_context:
            return "No conversation history available"

        summary_parts = []

        # Basic info
        intent = session_context.get('intent', 'unknown')
        summary_parts.append(f"Intent: {intent}")

        # Sentiment
        sentiment = session_context.get('sentiment', {})
        if sentiment:
            summary_parts.append(
                f"Sentiment: {sentiment.get('label', 'unknown')} "
                f"({sentiment.get('score', 0):.2f})"
            )

        # Entities
        entities = session_context.get('entities', [])
        if entities:
            entity_types = set(e.get('type', '') for e in entities)
            summary_parts.append(f"Entities found: {', '.join(entity_types)}")

        # Duration
        duration = session_context.get('duration_minutes', 0)
        summary_parts.append(f"Conversation duration: {duration} minutes")

        return " | ".join(summary_parts)

    def get_escalation_message(self, routing_info: Dict) -> str:
        """
        Generate message to show user during escalation

        Args:
            routing_info: Routing information

        Returns:
            Message string
        """
        message_parts = [
            "I understand this requires additional assistance.",
            f"I'm connecting you to our {routing_info['department']}.",
        ]

        wait_time = routing_info.get('estimated_wait_time')
        if wait_time:
            message_parts.append(
                f"Estimated wait time: {wait_time}."
            )

        message_parts.append(
            "A customer service representative will be with you shortly. "
            "Please hold while I transfer you."
        )

        return " ".join(message_parts)

    def get_escalation_stats(self) -> Dict:
        """Get escalation statistics"""
        if not self.escalation_history:
            return {
                'total_escalations': 0,
                'by_priority': {},
                'by_rule': {}
            }

        total = len(self.escalation_history)

        # Count by priority
        by_priority = {}
        for event in self.escalation_history:
            priority = event['priority']
            by_priority[priority] = by_priority.get(priority, 0) + 1

        # Count by rule
        by_rule = {}
        for event in self.escalation_history:
            for rule in event['triggered_rules']:
                by_rule[rule] = by_rule.get(rule, 0) + 1

        return {
            'total_escalations': total,
            'by_priority': by_priority,
            'by_rule': by_rule,
            'escalation_rate': f"{total} escalations"
        }


# Testing function
def test_escalation_manager():
    """Test escalation manager"""
    em = EscalationManager()

    test_contexts = [
        {
            'text': 'This is URGENT! I need help NOW!!!',
            'sentiment': {'score': -0.6, 'label': 'negative'},
            'urgency': {'score': 9.0, 'level': 'critical'},
            'intent': 'technical_support',
            'unresolved_turns': 1
        },
        {
            'text': 'I want to speak to a manager right now!',
            'sentiment': {'score': -0.8, 'label': 'very_negative'},
            'urgency': {'score': 6.0, 'level': 'high'},
            'intent': 'complaint',
            'unresolved_turns': 2
        },
        {
            'text': 'Can you help me with my order?',
            'sentiment': {'score': 0.1, 'label': 'neutral'},
            'urgency': {'score': 2.0, 'level': 'low'},
            'intent': 'order_status',
            'unresolved_turns': 0
        },
        {
            'text': 'I still don\'t understand. This is frustrating.',
            'sentiment': {'score': -0.4, 'label': 'negative'},
            'urgency': {'score': 4.0, 'level': 'medium'},
            'intent': 'product_inquiry',
            'unresolved_turns': 4
        }
    ]

    print("Escalation Manager Test")
    print("=" * 70)

    for i, context in enumerate(test_contexts, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {context['text']}")
        print(f"Sentiment: {context['sentiment']['label']} ({context['sentiment']['score']})")
        print(f"Urgency: {context['urgency']['level']} ({context['urgency']['score']})")
        print(f"Intent: {context['intent']}")

        should_escalate, info = em.should_escalate(context)

        print(f"\nShould Escalate: {should_escalate}")
        if should_escalate:
            print(f"Priority: {info['priority']}")
            print(f"Triggered Rules: {', '.join(info['triggered_rules'])}")
            print(f"Reasons: {'; '.join(info['reasons'])}")

            routing = em.get_routing_info(info, context)
            print(f"\nRouting Info:")
            print(f"  Department: {routing['department']}")
            print(f"  Queue: {routing['queue']}")
            print(f"  Wait Time: {routing['estimated_wait_time']}")
            print(f"\nEscalation Message:")
            print(f"  {em.get_escalation_message(routing)}")

        print("-" * 70)

    # Stats
    print("\nEscalation Statistics:")
    stats = em.get_escalation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_escalation_manager()
