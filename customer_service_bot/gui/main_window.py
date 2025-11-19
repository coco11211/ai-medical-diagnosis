"""
Main GUI Application for Customer Service Bot
Windows 11 compatible interface using tkinter/customtkinter
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from datetime import datetime
from typing import Optional, Dict
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerServiceBotGUI:
    """
    Main GUI application for customer service bot
    """

    def __init__(self, bot_controller):
        """
        Initialize GUI

        Args:
            bot_controller: CustomerServiceBot instance
        """
        self.bot = bot_controller
        self.root = tk.Tk()
        self.root.title("AI Customer Service Bot - Windows 11")
        self.root.geometry("1200x800")

        # Set Windows 11 theme
        self.setup_theme()

        # Current session
        self.current_session = None
        self.voice_enabled = False

        # Setup UI
        self.setup_ui()

        logger.info("GUI initialized")

    def setup_theme(self):
        """Setup Windows 11 style theme"""
        try:
            # Try to use modern theme
            import customtkinter as ctk
            ctk.set_appearance_mode("light")
            ctk.set_default_color_theme("blue")
        except ImportError:
            # Fallback to standard theme
            style = ttk.Style()
            style.theme_use('vista' if sys.platform == 'win32' else 'clam')

    def setup_ui(self):
        """Setup user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Left sidebar
        self.create_sidebar(main_frame)

        # Main chat area
        self.create_chat_area(main_frame)

        # Bottom input area
        self.create_input_area(main_frame)

        # Status bar
        self.create_status_bar()

    def create_sidebar(self, parent):
        """Create left sidebar with controls"""
        sidebar = ttk.Frame(parent, padding="5")
        sidebar.grid(row=0, column=0, rowspan=3, sticky=(tk.N, tk.S, tk.W))

        # Title
        title = ttk.Label(sidebar, text="Customer Service Bot",
                         font=('Segoe UI', 14, 'bold'))
        title.pack(pady=10)

        # New conversation button
        ttk.Button(sidebar, text="New Conversation",
                  command=self.new_conversation,
                  width=20).pack(pady=5)

        # Voice toggle
        self.voice_var = tk.BooleanVar()
        ttk.Checkbutton(sidebar, text="Enable Voice",
                       variable=self.voice_var,
                       command=self.toggle_voice).pack(pady=5)

        # Language selection
        ttk.Label(sidebar, text="Language:").pack(pady=(10, 2))
        self.language_var = tk.StringVar(value='en')
        languages = ['en', 'es', 'fr', 'de', 'ja', 'zh-CN']
        language_combo = ttk.Combobox(sidebar, textvariable=self.language_var,
                                     values=languages, state='readonly', width=18)
        language_combo.pack(pady=2)

        # Separator
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)

        # Session info
        ttk.Label(sidebar, text="Session Info",
                 font=('Segoe UI', 10, 'bold')).pack(pady=5)

        self.session_info = ttk.Label(sidebar, text="No active session",
                                     wraplength=150, justify='left')
        self.session_info.pack(pady=5)

        # Separator
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', pady=10)

        # Analytics button
        ttk.Button(sidebar, text="View Analytics",
                  command=self.open_analytics,
                  width=20).pack(pady=5)

        # Settings button
        ttk.Button(sidebar, text="Settings",
                  command=self.open_settings,
                  width=20).pack(pady=5)

        # Export button
        ttk.Button(sidebar, text="Export Conversation",
                  command=self.export_conversation,
                  width=20).pack(pady=5)

    def create_chat_area(self, parent):
        """Create main chat display area"""
        chat_frame = ttk.Frame(parent)
        chat_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.N, tk.S, tk.E, tk.W),
                       padx=(10, 0))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=35,
            font=('Segoe UI', 10),
            state='disabled',
            bg='#f5f5f5'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure tags for message styling
        self.chat_display.tag_config('user', foreground='#0066cc', font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config('bot', foreground='#00aa00', font=('Segoe UI', 10, 'bold'))
        self.chat_display.tag_config('system', foreground='#666666', font=('Segoe UI', 9, 'italic'))
        self.chat_display.tag_config('escalation', foreground='#cc0000', font=('Segoe UI', 10, 'bold'))

    def create_input_area(self, parent):
        """Create message input area"""
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=1, sticky=(tk.E, tk.W), padx=(10, 0), pady=(10, 0))

        # Input text area
        self.message_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=70,
            height=4,
            font=('Segoe UI', 10)
        )
        self.message_input.grid(row=0, column=0, sticky=(tk.E, tk.W))
        self.message_input.bind('<Return>', self.on_enter_key)
        self.message_input.bind('<Shift-Return>', lambda e: None)  # Allow Shift+Enter for new line

        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=1, padx=(5, 0))

        # Send button
        ttk.Button(button_frame, text="Send",
                  command=self.send_message,
                  width=10).pack(pady=2)

        # Voice input button
        ttk.Button(button_frame, text="🎤 Voice",
                  command=self.voice_input,
                  width=10).pack(pady=2)

        # Clear button
        ttk.Button(button_frame, text="Clear",
                  command=self.clear_input,
                  width=10).pack(pady=2)

        input_frame.columnconfigure(0, weight=1)

    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky=(tk.E, tk.W))

        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def new_conversation(self):
        """Start a new conversation"""
        if messagebox.askyesno("New Conversation",
                              "Start a new conversation? Current conversation will be saved."):
            # End current session if exists
            if self.current_session:
                self.bot.end_session(self.current_session)

            # Start new session
            self.current_session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.bot.start_session(self.current_session)

            # Clear chat
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state='disabled')

            # Update UI
            self.update_session_info()
            self.add_system_message("New conversation started")
            self.status_label.config(text=f"Session: {self.current_session}")

            logger.info(f"Started new conversation: {self.current_session}")

    def send_message(self):
        """Send user message"""
        message = self.message_input.get(1.0, tk.END).strip()

        if not message:
            return

        # Ensure session exists
        if not self.current_session:
            self.new_conversation()

        # Display user message
        self.add_message("You", message, 'user')

        # Clear input
        self.message_input.delete(1.0, tk.END)

        # Process message in background
        self.status_label.config(text="Processing...")
        threading.Thread(target=self._process_message, args=(message,), daemon=True).start()

    def _process_message(self, message: str):
        """Process message in background thread"""
        try:
            # Get response from bot
            response = self.bot.process_message(
                message,
                self.current_session,
                language=self.language_var.get()
            )

            # Display response on main thread
            self.root.after(0, self._display_response, response)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.root.after(0, self.add_system_message, f"Error: {str(e)}")
            self.root.after(0, self.status_label.config, {'text': 'Error occurred'})

    def _display_response(self, response: Dict):
        """Display bot response"""
        # Add bot message
        bot_text = response.get('response', 'I apologize, but I encountered an error.')
        self.add_message("Bot", bot_text, 'bot')

        # Check for escalation
        if response.get('escalation', {}).get('should_escalate', False):
            escalation_msg = response['escalation'].get('message', '')
            self.add_message("SYSTEM", escalation_msg, 'escalation')

        # Update status
        sentiment = response.get('sentiment', {}).get('label', 'neutral')
        intent = response.get('intent', {}).get('intent', 'unknown')
        self.status_label.config(
            text=f"Ready | Sentiment: {sentiment} | Intent: {intent}"
        )

    def add_message(self, sender: str, message: str, tag: str):
        """Add message to chat display"""
        self.chat_display.config(state='normal')

        timestamp = datetime.now().strftime("%H:%M:%S")

        self.chat_display.insert(tk.END, f"[{timestamp}] ", 'system')
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n")

        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

    def add_system_message(self, message: str):
        """Add system message"""
        self.chat_display.config(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] {message}\n", 'system')
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

    def on_enter_key(self, event):
        """Handle Enter key press"""
        if not event.state & 0x1:  # If Shift is not pressed
            self.send_message()
            return 'break'  # Prevent default newline

    def toggle_voice(self):
        """Toggle voice input/output"""
        self.voice_enabled = self.voice_var.get()
        status = "enabled" if self.voice_enabled else "disabled"
        self.add_system_message(f"Voice {status}")
        self.bot.set_voice_enabled(self.voice_enabled)

    def voice_input(self):
        """Get voice input"""
        if not self.voice_enabled:
            messagebox.showinfo("Voice Disabled", "Please enable voice first")
            return

        self.status_label.config(text="Listening...")
        threading.Thread(target=self._voice_input_thread, daemon=True).start()

    def _voice_input_thread(self):
        """Voice input in background thread"""
        try:
            text = self.bot.get_voice_input()

            if text:
                # Insert into message box
                self.root.after(0, self.message_input.insert, tk.END, text)
                self.root.after(0, self.status_label.config, {'text': 'Voice input received'})
            else:
                self.root.after(0, self.status_label.config, {'text': 'No speech detected'})

        except Exception as e:
            logger.error(f"Voice input error: {e}")
            self.root.after(0, self.status_label.config, {'text': 'Voice input failed'})

    def clear_input(self):
        """Clear message input"""
        self.message_input.delete(1.0, tk.END)

    def update_session_info(self):
        """Update session information display"""
        if self.current_session:
            stats = self.bot.get_session_stats(self.current_session)
            info_text = f"Session: {self.current_session}\n"
            info_text += f"Messages: {stats.get('message_count', 0)}\n"
            info_text += f"Duration: {stats.get('duration', 0)}s"
            self.session_info.config(text=info_text)

    def open_analytics(self):
        """Open analytics dashboard"""
        self.bot.open_analytics_dashboard()
        messagebox.showinfo("Analytics", "Analytics dashboard opened in browser")

    def open_settings(self):
        """Open settings dialog"""
        messagebox.showinfo("Settings", "Settings dialog - To be implemented")

    def export_conversation(self):
        """Export current conversation"""
        if not self.current_session:
            messagebox.showwarning("No Session", "No active conversation to export")
            return

        filename = f"conversation_{self.current_session}.json"
        self.bot.export_session(self.current_session, filename)
        messagebox.showinfo("Export", f"Conversation exported to {filename}")

    def run(self):
        """Run the GUI"""
        # Start with a new conversation
        self.new_conversation()

        # Welcome message
        welcome = ("Welcome to the AI Customer Service Bot!\n\n"
                  "Features:\n"
                  "• Natural language understanding\n"
                  "• Multi-language support\n"
                  "• Voice input/output\n"
                  "• Sentiment analysis\n"
                  "• Smart escalation\n\n"
                  "How can I help you today?")

        self.add_message("Bot", welcome, 'bot')

        # Run main loop
        logger.info("Starting GUI main loop")
        self.root.mainloop()


# For standalone testing
if __name__ == "__main__":
    # Mock bot controller for testing
    class MockBot:
        def start_session(self, session_id):
            pass

        def end_session(self, session_id):
            pass

        def process_message(self, message, session_id, language='en'):
            return {
                'response': f"Echo: {message}",
                'sentiment': {'label': 'neutral'},
                'intent': {'intent': 'general_inquiry'}
            }

        def set_voice_enabled(self, enabled):
            pass

        def get_voice_input(self):
            return "Test voice input"

        def get_session_stats(self, session_id):
            return {'message_count': 0, 'duration': 0}

        def open_analytics_dashboard(self):
            pass

        def export_session(self, session_id, filename):
            pass

    mock_bot = MockBot()
    app = CustomerServiceBotGUI(mock_bot)
    app.run()
