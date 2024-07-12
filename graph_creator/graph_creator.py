import plotly.graph_objects as go
import pandas as pd
from typing import List



class GraphBTC:


    def __init__(self, model: str, performance_data: pd.DataFrame, windows: List[str]):
        self.label = model
        self.performance_data = performance_data

        self.figure = go.Figure()
        self.annotations = []

        for color, metric in zip(["blue", "purple"], ["precision", "recall"]):
            for window in windows:
                self.add_line(metric, window, color)
                self.add_points(metric, window, color)
                
        for window in windows:
            self.annotate(window)
        
        self.set_layout()
        self.add_title()

        self.figure.update_layout(annotations=self.annotations)

    
    def add_line(self, metric: str, window: str, color: str) -> None:
        self.figure.add_trace(go.Scatter(
                x=self.performance_data["date"], 
                y=self.performance_data[f"{metric}_{window}"],
                mode="lines",
                name=f"{metric.capitalize()} Across All Time",
                opacity=0.7,
                line=dict(
                color=color,   
                width=4)))
        

    def add_points(self, metric: str, window: str, color: str) -> None:
        self.figure.add_trace(go.Scatter(
                        x=[self.performance_data["date"].iloc[0], self.performance_data["date"].iloc[-1]],
                        y=[self.performance_data[f"{metric}_{window}"].iloc[0], self.performance_data[f"{metric}_{window}"].iloc[-1]],
                        mode="markers",
                        marker=dict(color=color, size=10)))
        
    
    def set_layout(self) -> None:
        self.figure.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(0, 0, 0)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                showline=True,
                showticklabels=False,
            ),
            autosize=False,
            margin=dict(
                autoexpand=False,
                l=100,
                r=20,
                t=110,
            ),
            showlegend=False,
            plot_bgcolor='white'
        )

    
    def annotate(self, window: str) -> None:
        
        for y_trace, label, color in zip([self.performance_data[f"recall_{window}"], self.performance_data[f"precision_{window}"]], ["Recall", "Precision"], ["purple", "blue"]):
            # left side annotation
            self.annotations.append(dict(xref='paper', x=0.05, y=y_trace.iloc[0],
                                        xanchor='right', yanchor='middle',
                                        text=label + ' {}%'.format(round(100*y_trace.iloc[0])),
                                        font=dict(family='Arial',
                                                    size=14,
                                                    color=color),
                                        showarrow=False))
            # right side annotation
            self.annotations.append(dict(xref='paper', x=0.95, y=round(y_trace.iloc[-1], 2),
                                        xanchor='left', yanchor='middle',
                                        text='{}%'.format(round(100*y_trace.iloc[-1], 2)),
                                        font=dict(family='Arial',
                                                    size=14,
                                                    color=color),
                                        showarrow=False))
        

    def add_title(self) -> None:
        self.annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                    xanchor='left', yanchor='bottom',
                                    text=self.label.capitalize(),
                                    font=dict(family='Arial',
                                                size=30,
                                                color='rgb(37,37,37)'),
                                    showarrow=False))

    def get_graph(self) -> go.Figure:
        return self.figure