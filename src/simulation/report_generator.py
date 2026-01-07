"""
Simulation report generation.

Generates comprehensive reports from simulation runs including statistics,
visualizations, and comparisons. Supports multiple output formats
(markdown, HTML, PDF-ready).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

logger = logging.getLogger(__name__)


@dataclass
class SimulationMetrics:
    """
    Aggregated metrics from a simulation run.
    
    Attributes:
        run_id: Database run ID
        n_ticks: Number of simulation ticks
        total_trips: Total trips across all modes
        avg_energy: Average energy consumption per tick
        peak_energy: Peak energy consumption
        modal_shares: Dict of modal shares {mode: fraction}
        avg_congestion: Average congestion by mode
        peak_congestion: Peak congestion by mode
    """
    run_id: int
    n_ticks: int
    total_trips: float
    avg_energy: float
    peak_energy: float
    modal_shares: Dict[str, float]
    avg_congestion: Dict[str, float]
    peak_congestion: Dict[str, float]


class ReportGenerator:
    """
    Generate comprehensive simulation reports.
    
    Creates formatted reports with statistics, plots, and analysis
    from simulation runs. Supports markdown and HTML output.
    
    Example:
        >>> from simulation.report_generator import ReportGenerator
        >>> 
        >>> # Create report generator
        >>> reporter = ReportGenerator(output_dir="data/reports")
        >>> 
        >>> # Add simulation data
        >>> reporter.add_tick_data(df_ticks)
        >>> reporter.add_flow_data(df_flows)
        >>> 
        >>> # Generate report
        >>> report_path = reporter.generate_report(
        ...     run_id=1,
        ...     run_params=params,
        ...     format="markdown"
        ... )
    """
    
    def __init__(self, output_dir: str = "data/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for report outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.tick_data: Optional[pd.DataFrame] = None
        self.flow_data: Optional[pd.DataFrame] = None
        self.run_params: Optional[Dict[str, Any]] = None
        
        logger.info(f"ReportGenerator initialized: output_dir={output_dir}")
    
    def add_tick_data(self, df: pd.DataFrame) -> None:
        """
        Add tick-level metrics data.
        
        Args:
            df: DataFrame with columns: tick, car_congestion, ped_congestion,
                transit_congestion, energy_usage
        """
        self.tick_data = df.copy()
        logger.info(f"Added tick data: {len(df)} ticks")
    
    def add_flow_data(self, df: pd.DataFrame) -> None:
        """
        Add edge-mode flow data.
        
        Args:
            df: DataFrame with columns: tick, edge_id, mode, flow
        """
        self.flow_data = df.copy()
        logger.info(f"Added flow data: {len(df)} records")
    
    def compute_metrics(self) -> SimulationMetrics:
        """
        Compute aggregate metrics from data.
        
        Returns:
            SimulationMetrics object with computed statistics
        
        Raises:
            ValueError: If required data is missing
        """
        if self.tick_data is None:
            raise ValueError("No tick data available. Call add_tick_data() first.")
        
        # Basic metrics
        n_ticks = len(self.tick_data)
        avg_energy = float(self.tick_data["energy_usage"].mean())
        peak_energy = float(self.tick_data["energy_usage"].max())
        
        # Congestion metrics
        avg_congestion = {
            "car": float(self.tick_data["car_congestion"].mean()),
            "ped": float(self.tick_data["ped_congestion"].mean()),
            "transit": float(self.tick_data["transit_congestion"].mean()),
        }
        
        peak_congestion = {
            "car": float(self.tick_data["car_congestion"].max()),
            "ped": float(self.tick_data["ped_congestion"].max()),
            "transit": float(self.tick_data["transit_congestion"].max()),
        }
        
        # Flow-based metrics
        total_trips = 0.0
        modal_shares = {"car": 0.0, "ped": 0.0, "transit": 0.0}
        
        if self.flow_data is not None:
            # Total trips per mode
            mode_totals = self.flow_data.groupby("mode")["flow"].sum()
            total_trips = float(mode_totals.sum())
            
            if total_trips > 0:
                for mode in ["car", "ped", "transit"]:
                    if mode in mode_totals.index:
                        modal_shares[mode] = float(mode_totals[mode] / total_trips)
        
        return SimulationMetrics(
            run_id=0,  # Set by caller
            n_ticks=n_ticks,
            total_trips=total_trips,
            avg_energy=avg_energy,
            peak_energy=peak_energy,
            modal_shares=modal_shares,
            avg_congestion=avg_congestion,
            peak_congestion=peak_congestion,
        )
    
    def create_energy_plot(self, figsize: Tuple[int, int] = (10, 6)) -> matplotlib.figure.Figure:
        """
        Create energy consumption time series plot.
        
        Args:
            figsize: Figure size (width, height)
        
        Returns:
            Matplotlib figure
        """
        if self.tick_data is None:
            raise ValueError("No tick data available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            self.tick_data["tick"],
            self.tick_data["energy_usage"],
            marker="o",
            linewidth=2,
            markersize=6,
            color="darkred",
        )
        
        # Add mean line
        mean_energy = self.tick_data["energy_usage"].mean()
        ax.axhline(
            mean_energy,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_energy:.0f} kWh",
        )
        
        ax.set_xlabel("Tick", fontsize=12)
        ax.set_ylabel("Energy Consumption (kWh)", fontsize=12)
        ax.set_title("Energy Consumption Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_congestion_plot(self, figsize: Tuple[int, int] = (10, 6)) -> matplotlib.figure.Figure:
        """
        Create multi-mode congestion comparison plot.
        
        Args:
            figsize: Figure size (width, height)
        
        Returns:
            Matplotlib figure
        """
        if self.tick_data is None:
            raise ValueError("No tick data available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            self.tick_data["tick"],
            self.tick_data["car_congestion"],
            marker="o",
            label="Car",
            linewidth=2,
            markersize=5,
        )
        ax.plot(
            self.tick_data["tick"],
            self.tick_data["ped_congestion"],
            marker="s",
            label="Pedestrian",
            linewidth=2,
            markersize=5,
        )
        ax.plot(
            self.tick_data["tick"],
            self.tick_data["transit_congestion"],
            marker="^",
            label="Transit",
            linewidth=2,
            markersize=5,
        )
        
        ax.set_xlabel("Tick", fontsize=12)
        ax.set_ylabel("Mean Edge Flow", fontsize=12)
        ax.set_title("Multi-Modal Congestion Over Time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_modal_share_plot(
        self,
        metrics: SimulationMetrics,
        figsize: Tuple[int, int] = (8, 8),
    ) -> matplotlib.figure.Figure:
        """
        Create modal share pie chart.
        
        Args:
            metrics: SimulationMetrics with modal share data
            figsize: Figure size (width, height)
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        modes = ["car", "ped", "transit"]
        labels = ["Car", "Pedestrian", "Transit"]
        sizes = [metrics.modal_shares[m] * 100 for m in modes]
        colors = ["#ff9999", "#66b3ff", "#99ff99"]
        explode = (0.05, 0, 0)  # Explode car slice
        
        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 12},
        )
        
        ax.set_title("Modal Share Distribution", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        return fig
    
    def create_summary_table(self, metrics: SimulationMetrics) -> str:
        """
        Create formatted summary table in markdown.
        
        Args:
            metrics: SimulationMetrics object
        
        Returns:
            Markdown-formatted table string
        """
        lines = []
        
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Run ID | {metrics.run_id} |")
        lines.append(f"| Simulation Ticks | {metrics.n_ticks} |")
        lines.append(f"| Total Trips | {metrics.total_trips:,.0f} |")
        lines.append(f"| Avg Energy (kWh/tick) | {metrics.avg_energy:,.0f} |")
        lines.append(f"| Peak Energy (kWh) | {metrics.peak_energy:,.0f} |")
        lines.append(f"| Car Modal Share | {metrics.modal_shares['car']:.1%} |")
        lines.append(f"| Pedestrian Modal Share | {metrics.modal_shares['ped']:.1%} |")
        lines.append(f"| Transit Modal Share | {metrics.modal_shares['transit']:.1%} |")
        lines.append(f"| Avg Car Congestion | {metrics.avg_congestion['car']:.2f} |")
        lines.append(f"| Avg Ped Congestion | {metrics.avg_congestion['ped']:.2f} |")
        lines.append(f"| Avg Transit Congestion | {metrics.avg_congestion['transit']:.2f} |")
        
        return "\n".join(lines)
    
    def generate_report(
        self,
        run_id: int,
        run_params: Optional[Dict[str, Any]] = None,
        format: str = "markdown",
        include_plots: bool = True,
    ) -> str:
        """
        Generate comprehensive simulation report.
        
        Args:
            run_id: Simulation run ID
            run_params: Run configuration parameters
            format: Output format ("markdown" or "html")
            include_plots: Whether to generate and include plots
        
        Returns:
            Path to generated report file
        
        Example:
            >>> report_path = reporter.generate_report(
            ...     run_id=1,
            ...     run_params={"seed": 42, "ticks": 10},
            ...     format="markdown",
            ...     include_plots=True
            ... )
            >>> print(f"Report saved to: {report_path}")
        """
        logger.info(f"Generating {format} report for run_id={run_id}")
        
        # Compute metrics
        metrics = self.compute_metrics()
        metrics.run_id = run_id
        
        # Create output subdirectory for this run
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Generate plots
        plot_paths = {}
        if include_plots:
            logger.info("Generating plots...")
            
            # Energy plot
            fig_energy = self.create_energy_plot()
            energy_path = run_dir / "energy_over_time.png"
            fig_energy.savefig(energy_path, dpi=150, bbox_inches="tight")
            plt.close(fig_energy)
            plot_paths["energy"] = energy_path.name
            
            # Congestion plot
            fig_congestion = self.create_congestion_plot()
            congestion_path = run_dir / "congestion_over_time.png"
            fig_congestion.savefig(congestion_path, dpi=150, bbox_inches="tight")
            plt.close(fig_congestion)
            plot_paths["congestion"] = congestion_path.name
            
            # Modal share plot
            fig_modal = self.create_modal_share_plot(metrics)
            modal_path = run_dir / "modal_share.png"
            fig_modal.savefig(modal_path, dpi=150, bbox_inches="tight")
            plt.close(fig_modal)
            plot_paths["modal"] = modal_path.name
        
        # Generate report
        if format == "markdown":
            report_path = self._generate_markdown_report(
                run_id, metrics, run_params, plot_paths, run_dir
            )
        elif format == "html":
            report_path = self._generate_html_report(
                run_id, metrics, run_params, plot_paths, run_dir
            )
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(
        self,
        run_id: int,
        metrics: SimulationMetrics,
        run_params: Optional[Dict[str, Any]],
        plot_paths: Dict[str, str],
        output_dir: Path,
    ) -> Path:
        """Generate markdown report."""
        lines = []
        
        # Header
        lines.append(f"# Simulation Report - Run {run_id}")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        lines.append("## Summary Metrics\n")
        lines.append(self.create_summary_table(metrics))
        lines.append("\n")
        
        # Configuration
        if run_params:
            lines.append("## Configuration\n")
            lines.append("```json")
            import json
            lines.append(json.dumps(run_params, indent=2))
            lines.append("```\n")
        
        # Plots
        if plot_paths:
            lines.append("## Visualizations\n")
            
            if "energy" in plot_paths:
                lines.append(f"### Energy Consumption\n")
                lines.append(f"![Energy Over Time]({plot_paths['energy']})\n")
            
            if "congestion" in plot_paths:
                lines.append(f"### Congestion Analysis\n")
                lines.append(f"![Congestion Over Time]({plot_paths['congestion']})\n")
            
            if "modal" in plot_paths:
                lines.append(f"### Modal Share\n")
                lines.append(f"![Modal Share]({plot_paths['modal']})\n")
        
        # Analysis
        lines.append("## Analysis\n")
        lines.append(self._generate_analysis_text(metrics))
        
        # Write file
        report_path = output_dir / f"report_run_{run_id}.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        
        return report_path
    
    def _generate_html_report(
        self,
        run_id: int,
        metrics: SimulationMetrics,
        run_params: Optional[Dict[str, Any]],
        plot_paths: Dict[str, str],
        output_dir: Path,
    ) -> Path:
        """Generate HTML report."""
        # Convert markdown to HTML structure
        lines = []
        
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"<title>Simulation Report - Run {run_id}</title>")
        lines.append("<style>")
        lines.append(self._get_html_style())
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")
        
        lines.append(f"<h1>Simulation Report - Run {run_id}</h1>")
        lines.append(f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary
        lines.append("<h2>Summary Metrics</h2>")
        lines.append(self._metrics_to_html_table(metrics))
        
        # Plots
        if plot_paths:
            lines.append("<h2>Visualizations</h2>")
            for plot_name, plot_file in plot_paths.items():
                lines.append(f"<img src='{plot_file}' alt='{plot_name}' style='max-width: 800px;'>")
        
        lines.append("</body>")
        lines.append("</html>")
        
        # Write file
        report_path = output_dir / f"report_run_{run_id}.html"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        
        return report_path
    
    def _generate_analysis_text(self, metrics: SimulationMetrics) -> str:
        """Generate analysis insights text."""
        lines = []
        
        # Energy analysis
        lines.append(f"**Energy Consumption**: The simulation averaged "
                    f"{metrics.avg_energy:,.0f} kWh per tick with a peak of "
                    f"{metrics.peak_energy:,.0f} kWh.")
        
        # Modal analysis
        dominant_mode = max(metrics.modal_shares.items(), key=lambda x: x[1])
        lines.append(f"\n**Modal Distribution**: {dominant_mode[0].capitalize()} "
                    f"was the dominant mode at {dominant_mode[1]:.1%} of total trips.")
        
        # Congestion analysis
        max_cong_mode = max(metrics.avg_congestion.items(), key=lambda x: x[1])
        lines.append(f"\n**Congestion**: {max_cong_mode[0].capitalize()} showed "
                    f"the highest average congestion at {max_cong_mode[1]:.2f} "
                    f"mean edge flow.")
        
        return "\n".join(lines)
    
    def _metrics_to_html_table(self, metrics: SimulationMetrics) -> str:
        """Convert metrics to HTML table."""
        rows = [
            ("Run ID", metrics.run_id),
            ("Simulation Ticks", metrics.n_ticks),
            ("Total Trips", f"{metrics.total_trips:,.0f}"),
            ("Avg Energy (kWh/tick)", f"{metrics.avg_energy:,.0f}"),
            ("Peak Energy (kWh)", f"{metrics.peak_energy:,.0f}"),
        ]
        
        html = "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        for label, value in rows:
            html += f"<tr><td>{label}</td><td>{value}</td></tr>"
        html += "</table>"
        
        return html
    
    def _get_html_style(self) -> str:
        """Get CSS style for HTML reports."""
        return """
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        table { border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        img { margin: 20px 0; border: 1px solid #ddd; }
        .timestamp { color: #7f8c8d; font-style: italic; }
        """


def create_quick_report(
    tick_csv: str,
    edge_csv: str,
    run_id: int,
    output_dir: str = "data/reports",
) -> str:
    """
    Quick report generation from CSV files.
    
    Convenience function for generating reports directly from CSV exports.
    
    Args:
        tick_csv: Path to tick-level CSV
        edge_csv: Path to edge-mode CSV
        run_id: Run ID
        output_dir: Output directory
    
    Returns:
        Path to generated report
    
    Example:
        >>> report_path = create_quick_report(
        ...     "data/run_1_ticks.csv",
        ...     "data/run_1_edge_mode.csv",
        ...     run_id=1
        ... )
    """
    reporter = ReportGenerator(output_dir=output_dir)
    
    # Load data
    df_ticks = pd.read_csv(tick_csv)
    df_edges = pd.read_csv(edge_csv)
    
    reporter.add_tick_data(df_ticks)
    reporter.add_flow_data(df_edges)
    
    # Generate report
    return reporter.generate_report(run_id=run_id, include_plots=True)