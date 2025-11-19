"""
Interactive Dashboard
Web-based analytics dashboard for students and tutors using Dash/Plotly
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .models import Student, Concept, LearningSession
from .database import DatabaseManager
from .progress_tracker import ProgressTracker
from .knowledge_tracer import KnowledgeTracer


class TutoringDashboard:
    """
    Interactive web dashboard for the tutoring platform
    """

    def __init__(self, database: DatabaseManager):
        """
        Initialize dashboard

        Args:
            database: DatabaseManager instance
        """
        self.database = database
        self.progress_tracker = ProgressTracker()
        self.knowledge_tracer = KnowledgeTracer()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            title="AI Tutoring Platform Dashboard",
            suppress_callback_exceptions=True
        )

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1("AI Tutoring Platform Dashboard",
                       style={'color': '#2c3e50', 'textAlign': 'center'}),
                html.Hr()
            ]),

            # Student selector
            html.Div([
                html.Label("Select Student:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='student-selector',
                    options=[],
                    value=None,
                    style={'width': '50%'}
                )
            ], style={'padding': '20px'}),

            # Main dashboard
            html.Div(id='dashboard-content', children=[
                # Overall stats
                html.Div(id='stats-cards', children=[
                    self._create_stat_card("Total Concepts", "0", "📚"),
                    self._create_stat_card("Mastered", "0", "✅"),
                    self._create_stat_card("Accuracy", "0%", "🎯"),
                    self._create_stat_card("Study Streak", "0 days", "🔥")
                ], style={
                    'display': 'flex',
                    'justifyContent': 'space-around',
                    'marginBottom': '30px',
                    'flexWrap': 'wrap'
                }),

                # Charts row 1
                html.Div([
                    html.Div([
                        html.H3("Mastery by Subject"),
                        dcc.Graph(id='mastery-by-subject')
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("Learning Progress Over Time"),
                        dcc.Graph(id='progress-over-time')
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginBottom': '30px'}),

                # Charts row 2
                html.Div([
                    html.Div([
                        html.H3("Time Distribution by Subject"),
                        dcc.Graph(id='time-distribution')
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("Concept Mastery Levels"),
                        dcc.Graph(id='mastery-levels')
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ], style={'marginBottom': '30px'}),

                # Weak areas and strengths
                html.Div([
                    html.Div([
                        html.H3("Areas for Improvement", style={'color': '#e74c3c'}),
                        html.Div(id='weak-areas')
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    html.Div([
                        html.H3("Strengths", style={'color': '#27ae60'}),
                        html.Div(id='strengths')
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'})
                ], style={'marginBottom': '30px'}),

                # Recommendations
                html.Div([
                    html.H3("Personalized Recommendations"),
                    html.Div(id='recommendations')
                ], style={'marginBottom': '30px'})
            ]),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ], style={'padding': '20px', 'fontFamily': 'Segoe UI, Arial, sans-serif'})

    def _create_stat_card(self, title: str, value: str, icon: str) -> html.Div:
        """Create a stat card component"""
        return html.Div([
            html.Div(icon, style={'fontSize': '48px', 'textAlign': 'center'}),
            html.Div(value, style={
                'fontSize': '32px',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'margin': '10px 0'
            }),
            html.Div(title, style={
                'fontSize': '16px',
                'color': '#7f8c8d',
                'textAlign': 'center'
            })
        ], style={
            'width': '200px',
            'padding': '20px',
            'margin': '10px',
            'backgroundColor': '#ecf0f1',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })

    def setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output('student-selector', 'options'),
            Input('interval-component', 'n_intervals')
        )
        def update_student_list(n):
            """Update list of students"""
            students = self.database.get_all_students()
            return [
                {'label': f"{s.name} ({s.email})", 'value': s.id}
                for s in students
            ]

        @self.app.callback(
            [
                Output('stats-cards', 'children'),
                Output('mastery-by-subject', 'figure'),
                Output('progress-over-time', 'figure'),
                Output('time-distribution', 'figure'),
                Output('mastery-levels', 'figure'),
                Output('weak-areas', 'children'),
                Output('strengths', 'children'),
                Output('recommendations', 'children')
            ],
            Input('student-selector', 'value')
        )
        def update_dashboard(student_id):
            """Update all dashboard components"""
            if not student_id:
                return self._empty_dashboard()

            student = self.database.get_student(student_id)
            if not student:
                return self._empty_dashboard()

            concepts = self.database.get_all_concepts()
            sessions = self.database.get_sessions_by_student(student_id, limit=100)

            # Generate report
            report = self.progress_tracker.generate_progress_report(
                student, concepts, sessions
            )

            # Stats cards
            summary = report['overall_summary']
            stats_cards = [
                self._create_stat_card(
                    "Total Concepts",
                    str(summary['total_concepts']),
                    "📚"
                ),
                self._create_stat_card(
                    "Mastered",
                    f"{summary['mastered_concepts']}/{summary['total_concepts']}",
                    "✅"
                ),
                self._create_stat_card(
                    "Accuracy",
                    f"{summary['accuracy']*100:.1f}%",
                    "🎯"
                ),
                self._create_stat_card(
                    "Study Streak",
                    f"{summary['streak_days']} days",
                    "🔥"
                )
            ]

            # Mastery by subject chart
            subject_data = report['subject_breakdown']
            mastery_fig = self._create_mastery_by_subject_chart(subject_data)

            # Progress over time chart
            progress_fig = self._create_progress_over_time_chart(sessions)

            # Time distribution chart
            time_fig = self._create_time_distribution_chart(report['time_distribution'])

            # Mastery levels chart
            mastery_levels_fig = self._create_mastery_levels_chart(student)

            # Weak areas
            weak_areas = self._create_weak_areas_list(report['weak_areas'])

            # Strengths
            strengths = self._create_strengths_list(report['strengths'])

            # Recommendations
            recommendations = self._create_recommendations_list(report['recommendations'])

            return (
                stats_cards,
                mastery_fig,
                progress_fig,
                time_fig,
                mastery_levels_fig,
                weak_areas,
                strengths,
                recommendations
            )

    def _create_mastery_by_subject_chart(self, subject_data: Dict) -> go.Figure:
        """Create mastery by subject bar chart"""
        if not subject_data:
            return self._empty_figure("No subject data available")

        subjects = list(subject_data.keys())
        mastery_scores = [data['average_mastery'] * 100 for data in subject_data.values()]

        fig = go.Figure(data=[
            go.Bar(
                x=subjects,
                y=mastery_scores,
                marker_color='#3498db',
                text=[f"{score:.1f}%" for score in mastery_scores],
                textposition='auto'
            )
        ])

        fig.update_layout(
            yaxis_title="Mastery %",
            yaxis_range=[0, 100],
            showlegend=False,
            template='plotly_white'
        )

        return fig

    def _create_progress_over_time_chart(self, sessions: List[LearningSession]) -> go.Figure:
        """Create progress over time line chart"""
        if not sessions:
            return self._empty_figure("No session data available")

        # Sort by date
        sessions_sorted = sorted(sessions, key=lambda s: s.start_time)

        dates = [s.start_time for s in sessions_sorted]
        performance = [s.performance_score for s in sessions_sorted]

        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=performance,
                mode='lines+markers',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=8)
            )
        ])

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Performance Score",
            yaxis_range=[0, 100],
            showlegend=False,
            template='plotly_white'
        )

        return fig

    def _create_time_distribution_chart(self, time_data: Dict[str, int]) -> go.Figure:
        """Create time distribution pie chart"""
        if not time_data:
            return self._empty_figure("No time data available")

        subjects = list(time_data.keys())
        times = [t / 3600 for t in time_data.values()]  # Convert to hours

        fig = go.Figure(data=[
            go.Pie(
                labels=subjects,
                values=times,
                hole=0.3,
                textinfo='label+percent'
            )
        ])

        fig.update_layout(
            showlegend=True,
            template='plotly_white'
        )

        return fig

    def _create_mastery_levels_chart(self, student: Student) -> go.Figure:
        """Create mastery levels distribution"""
        from .models import MasteryLevel

        levels = [0, 0, 0, 0, 0]  # NOT_LEARNED, LEARNING, FAMILIAR, PROFICIENT, MASTERED

        for state in student.knowledge_states.values():
            levels[state.mastery_level.value] += 1

        fig = go.Figure(data=[
            go.Bar(
                x=['Not Learned', 'Learning', 'Familiar', 'Proficient', 'Mastered'],
                y=levels,
                marker_color=['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#27ae60']
            )
        ])

        fig.update_layout(
            yaxis_title="Number of Concepts",
            showlegend=False,
            template='plotly_white'
        )

        return fig

    def _create_weak_areas_list(self, weak_areas: List[Dict]) -> html.Div:
        """Create weak areas list"""
        if not weak_areas:
            return html.Div("No weak areas identified! Great job! 🎉")

        items = []
        for area in weak_areas[:5]:
            items.append(html.Div([
                html.Div(f"• {area['concept']} ({area['subject']})",
                        style={'fontWeight': 'bold'}),
                html.Div(f"  Mastery: {area['mastery']*100:.1f}% | "
                        f"Accuracy: {area['accuracy']*100:.1f}% | "
                        f"Attempts: {area['attempts']}",
                        style={'color': '#7f8c8d', 'fontSize': '14px'})
            ], style={'marginBottom': '10px'}))

        return html.Div(items)

    def _create_strengths_list(self, strengths: List[Dict]) -> html.Div:
        """Create strengths list"""
        if not strengths:
            return html.Div("Keep practicing to build your strengths!")

        items = []
        for strength in strengths[:5]:
            items.append(html.Div([
                html.Div(f"• {strength['concept']} ({strength['subject']})",
                        style={'fontWeight': 'bold'}),
                html.Div(f"  Mastery: {strength['mastery']*100:.1f}% | "
                        f"Accuracy: {strength['accuracy']*100:.1f}% | "
                        f"Attempts: {strength['attempts']}",
                        style={'color': '#7f8c8d', 'fontSize': '14px'})
            ], style={'marginBottom': '10px'}))

        return html.Div(items)

    def _create_recommendations_list(self, recommendations: List[str]) -> html.Div:
        """Create recommendations list"""
        if not recommendations:
            return html.Div("No recommendations at this time.")

        items = [
            html.Div(f"• {rec}", style={'marginBottom': '10px'})
            for rec in recommendations
        ]

        return html.Div(items)

    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(template='plotly_white')
        return fig

    def _empty_dashboard(self):
        """Return empty dashboard components"""
        empty_fig = self._empty_figure("Select a student to view dashboard")

        return (
            [self._create_stat_card("", "0", "")],
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            html.Div("Select a student"),
            html.Div("Select a student"),
            html.Div("Select a student")
        )

    def run(self, debug: bool = False, port: int = 8050):
        """
        Run the dashboard server

        Args:
            debug: Enable debug mode
            port: Port to run on
        """
        print(f"Starting AI Tutoring Dashboard on http://127.0.0.1:{port}")
        self.app.run_server(debug=debug, port=port)
