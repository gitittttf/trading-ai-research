import asyncio
import aiohttp
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, DataTable, Label, ProgressBar, TabbedContent, TabPane
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.binding import Binding
from textual.screen import Screen, ModalScreen
from textual.message import Message
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.align import Align

# --- THEME COLORS ---
BG_COLOR = "#0a0e14"
GREEN = "#00ff87"
RED = "#ff3333"
AMBER = "#ffaa00"
GREY = "#3a3f4b"
TEXT_PRIMARY = "#cdd6f4"
TEXT_SECONDARY = "#a6adc8"
YELLOW = "#f9e2af"
PURPLE = "#cba6f7"

class PositionDetailScreen(ModalScreen):
    def __init__(self, position: dict):
        super().__init__()
        self.position = position

    def compose(self) -> ComposeResult:
        p = self.position
        lines = [
            f"[bold {YELLOW}]PAIR:[/] {p.get('pair')}",
            f"[bold {YELLOW}]DIRECTION:[/] {'LONG' if p.get('trade_dir') == 1 else 'SHORT'}",
            f"[bold {YELLOW}]ENTRY TIMESTAMP:[/] {datetime.fromtimestamp(p.get('entry_ts', 0)/1000).strftime('%Y-%m-%d %H:%M:%S')}",
            f"[bold {YELLOW}]ENTRY P1:[/] {p.get('entry_p1'):.4f}",
            f"[bold {YELLOW}]ENTRY P2:[/] {p.get('entry_p2'):.4f}",
            f"[bold {YELLOW}]ENTRY Z-SCORE:[/] {p.get('entry_z'):.2f}",
            f"[bold {YELLOW}]ENTRY STD:[/] {p.get('entry_std'):.6f}",
            f"[bold {YELLOW}]STOP DISTANCE:[/] {p.get('delta_z_stop'):.2f}",
            f"[bold {YELLOW}]ADJUSTED ALPHA:[/] {p.get('adjusted_alpha'):.4f}",
            f"[bold {YELLOW}]UNCERTAINTY:[/] {p.get('uncertainty'):.4f}",
            f"[bold {YELLOW}]ENTRY ADF:[/] {p.get('entry_adf'):.4f}",
            f"[bold {YELLOW}]ENTRY VOLATILITY:[/] {p.get('entry_volatility'):.6f}",
            f"[bold {YELLOW}]Z-MOMENTUM:[/] {p.get('z_momentum'):.4f}",
            f"[bold {YELLOW}]ALLOCATED MARGIN:[/] ${p.get('allocated_margin'):.2f}",
            f"[bold {YELLOW}]SIZE P1:[/] {p.get('size_p1'):.4f}",
            f"[bold {YELLOW}]SIZE P2:[/] {p.get('size_p2'):.4f}",
            f"[bold {YELLOW}]META PROB:[/] {p.get('meta_prob'):.2f}",
        ]
        with Container(id="modal_container"):
            yield Static(Panel("\n".join(lines), title="[bold]POSITION DETAILS[/]", border_style=PURPLE))
            yield Label("Press ESC or ENTER to close", id="modal_footer")

    def on_key(self, event):
        if event.key == "escape" or event.key == "enter":
            self.dismiss()

class LeadQuantDashboard(App):
    TITLE = "LEAD QUANT PAPER TRADER"
    CSS = f"""
    Screen {{
        background: {BG_COLOR};
        color: {TEXT_PRIMARY};
    }}

    #header_stats {{
        height: 5;
        border-bottom: solid {GREY};
        padding: 1 2;
        background: #0d121a;
    }}

    .stat-label {{
        color: {TEXT_SECONDARY};
    }}

    .stat-value {{
        text-style: bold;
        margin-right: 2;
    }}

    .value-pos {{ color: {GREEN}; }}
    .value-neg {{ color: {RED}; }}
    .value-warn {{ color: {AMBER}; }}
    .value-neutral {{ color: {TEXT_PRIMARY}; }}

    #main_content {{
        height: 1fr;
    }}

    TabbedContent {{
        height: 1fr;
    }}

    TabPane {{
        padding: 1 2;
    }}

    DataTable {{
        height: 1fr;
        border: none;
        background: transparent;
    }}

    #overlay_offline {{
        display: none;
        background: rgba(255, 0, 0, 0.4);
        color: white;
        content-align: center middle;
        text-style: bold;
        width: 100%;
        height: 100%;
        layer: overlay;
    }}

    #overlay_paused {{
        display: none;
        background: rgba(0, 0, 0, 0.85);
        color: yellow;
        content-align: center middle;
        text-style: bold;
        width: 100%;
        height: 100%;
        layer: overlay;
    }}

    .radar-cell {{
        border: solid {GREY};
        padding: 1;
        height: 7;
        margin: 1;
    }}

    Footer {{
        background: #0d121a;
        color: {TEXT_SECONDARY};
    }}

    #modal_container {{
        background: rgba(0, 0, 0, 0.8);
        content-align: center middle;
    }}

    #modal_container Static {{
        width: 60;
        height: 25;
        background: {BG_COLOR};
    }}

    #modal_footer {{
        text-align: center;
        color: {GREY};
        margin-top: 1;
    }}
    """

    BINDINGS = [
        Binding("1", "switch_tab('positions')", "Positions", show=False),
        Binding("2", "switch_tab('equity')", "Equity", show=False),
        Binding("3", "switch_tab('blotter')", "Blotter", show=False),
        Binding("4", "switch_tab('radar')", "Radar", show=False),
        Binding("5", "switch_tab('risk')", "Risk", show=False),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("r", "refresh_now", "Refresh Now"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.state = {}
        self.paused = False
        self.error_count = 0
        self.api_url = "http://127.0.0.1:8000/state"
        self.equity_history = []

    def compose(self) -> ComposeResult:
        yield Static("CORE OFFLINE - ATTEMPTING RECONNECT...", id="overlay_offline")
        yield Static("REFRESH PAUSED", id="overlay_paused")
        
        with Vertical(id="header_stats"):
            with Horizontal():
                yield Label(Text("EQUITY: ", style=TEXT_SECONDARY), classes="stat-label")
                yield Label("$0.00", id="val_equity", classes="stat-value")
                
                yield Label(Text("PNL: ", style=TEXT_SECONDARY), classes="stat-label")
                yield Label("$0.00 (0.0%)", id="val_pnl", classes="stat-value")
                
                yield Label(Text("BREAKER: ", style=TEXT_SECONDARY), classes="stat-label")
                yield Label("0.00", id="val_breaker", classes="stat-value")
                
                yield Label(Text("MSI: ", style=TEXT_SECONDARY), classes="stat-label")
                yield Label("0.00 (NORMAL)", id="val_msi", classes="stat-value")
            
            with Horizontal():
                yield Label(Text("TIME: ", style=TEXT_SECONDARY), classes="stat-label")
                yield Label("--:--:--", id="val_time", classes="stat-value")
                yield Label(Text("STATUS: ", style=TEXT_SECONDARY), classes="stat-label")
                yield Label("CONNECTING...", id="val_status", classes="stat-value")

        with Container(id="main_content"):
            with TabbedContent(initial="positions", id="tabs"):
                with TabPane("Positions", id="positions"):
                    yield DataTable(id="table_positions")
                with TabPane("Equity", id="equity"):
                    yield Static("Equity performance tracking...", id="equity_chart")
                    yield DataTable(id="table_equity_stats")
                with TabPane("Blotter", id="blotter"):
                    yield DataTable(id="table_blotter")
                with TabPane("Radar", id="radar"):
                    yield Grid(id="radar_grid")
                with TabPane("Risk", id="risk"):
                    with Vertical():
                        yield Label("Market Stress Index (MSI)")
                        yield ProgressBar(total=1.0, id="pb_msi")
                        yield Static("", id="risk_details")

        yield Footer()

    async def on_mount(self) -> None:
        self.setup_tables()
        self.set_interval(1.0, self.update_state)

    def setup_tables(self):
        # Positions Table
        pos_table = self.query_one("#table_positions", DataTable)
        pos_table.add_columns("Pair", "Dir", "Entry Z", "Duration", "Meta Prob", "Unreal PnL")
        pos_table.cursor_type = "row"

        # Blotter Table
        blotter_table = self.query_one("#table_blotter", DataTable)
        blotter_table.add_columns("Pair", "Exit Reason", "Entry Price", "Exit Price", "Net PnL", "Return%")
        
        # Equity Stats Table
        eq_table = self.query_one("#table_equity_stats", DataTable)
        eq_table.add_columns("Metric", "Value")

    async def update_state(self):
        if self.paused:
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, timeout=0.8) as response:
                    if response.status == 200:
                        self.state = await response.json()
                        self.error_count = 0
                        self.query_one("#overlay_offline").styles.display = "none"
                        
                        eq = self.state.get("equity", 1000.0)
                        self.equity_history.append(eq)
                        if len(self.equity_history) > 50:
                            self.equity_history.pop(0)
                            
                        self.refresh_ui()
                    else:
                        self.handle_error()
        except asyncio.CancelledError:
            pass
        except Exception:
            self.handle_error()

    def handle_error(self):
        self.error_count += 1
        if self.error_count >= 3:
            self.query_one("#overlay_offline").styles.display = "block"
            self.query_one("#val_status").update("OFFLINE")
            self.query_one("#val_status").styles.color = RED

    def refresh_ui(self):
        s = self.state
        # Header
        eq = s.get("equity", 0.0)
        pnl = s.get("pnl", 0.0)
        pnl_pct = (pnl / (s.get("starting_equity", 1000.0))) * 100
        
        self.query_one("#val_equity").update(f"${eq:,.2f}")
        
        pnl_text = f"${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        pnl_label = self.query_one("#val_pnl")
        pnl_label.update(pnl_text)
        pnl_label.styles.color = GREEN if pnl >= 0 else RED
        
        breaker = s.get("breaker_penalty", 0.0)
        breaker_label = self.query_one("#val_breaker")
        breaker_label.update(f"{breaker:.2f}")
        breaker_label.styles.color = PURPLE if breaker > 0 else TEXT_PRIMARY
        
        msi_rate = s.get("msi_breakdown_rate", 0.0)
        msi_text = "STRESSED" if s.get("msi_stressed") else "NORMAL"
        self.query_one("#val_msi").update(f"{msi_rate:.2f} ({msi_text})")
        self.query_one("#val_msi").styles.color = AMBER if s.get("msi_stressed") else TEXT_PRIMARY
        
        self.query_one("#val_time").update(datetime.now().strftime("%H:%M:%S"))
        self.query_one("#val_status").update("LIVE")
        self.query_one("#val_status").styles.color = GREEN

        # Positions Table
        self.update_positions_table()
        
        # Blotter Table
        self.update_blotter_table()
        
        # Radar Grid
        self.update_radar_grid()
        
        # Risk Panel
        self.query_one("#pb_msi").progress = msi_rate
        risk_details = f"""
        Portfolio Margin Usage: {s.get('active_trades', 0)} / 5 Max Pairs
        Current Breaker Penalty: {breaker:.2f}
        MSI Rate (Last 24h): {msi_rate:.2f}
        System Timestamp: {datetime.fromtimestamp(s.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}
        """
        self.query_one("#risk_details").update(risk_details)
        
        self.update_equity_tab()

    def update_equity_tab(self):
        if len(self.equity_history) < 2:
            self.query_one("#equity_chart").update("Waiting for more data points...")
            return
        
        # Sparkline (unicode block characters)
        hist = self.equity_history
        min_val = min(hist)
        max_val = max(hist)
        
        if max_val - min_val == 0:
            spark = "▁" * len(hist)
        else:
            blocks = " ▂▃▄▅▆▇█"
            spark = "".join(blocks[int((v - min_val) / (max_val - min_val + 1e-8) * 7)] for v in hist)
        
        self.query_one("#equity_chart").update(f"Equity Sparkline (last {len(hist)} samples):\n[bold {GREEN}]{spark}[/]")
        
        # Stats
        eq_table = self.query_one("#table_equity_stats", DataTable)
        eq_table.clear()
        
        returns = [hist[i] - hist[i-1] for i in range(1, len(hist))]
        if returns:
            total_pnl = hist[-1] - hist[0]
            # Simple Max Drawdown calculation
            peak = hist[0]
            max_dd = 0
            for v in hist:
                if v > peak: peak = v
                dd = peak - v
                if dd > max_dd: max_dd = dd
            
            eq_table.add_row("Max Drawdown", f"${max_dd:.2f}")
            eq_table.add_row("Total PnL", f"${total_pnl:+.2f}")
            
            downside = [r for r in returns if r < 0]
            if downside:
                sortino = np.mean(returns) / (np.std(downside) + 1e-8)
                eq_table.add_row("Sortino (est)", f"{sortino:.2f}")

    def update_positions_table(self):
        table = self.query_one("#table_positions", DataTable)
        table.clear()
        positions = self.state.get("open_positions", [])
        if not positions:
            table.add_row("No open positions", "", "", "", "", "")
            return
            
        for pos in positions:
            pair = pos.get("pair", "UNK")
            dir_str = "LONG" if pos.get("trade_dir") == 1 else "SHORT"
            z = f"{pos.get('entry_z', 0.0):.2f}"
            
            # Duration
            entry_ts = pos.get("entry_ts", time.time()*1000)
            duration_ms = (time.time()*1000) - entry_ts
            duration_min = int(duration_ms / 60000)
            dur_str = f"{duration_min}m"
            
            prob = f"{pos.get('meta_prob', 0.0):.2f}"
            
            # Unrealized PnL (Approximation if we don't have current prices here)
            # For now just show "ACTIVE"
            pnl = "N/A" # In a real app, API would send current PnL
            
            table.add_row(pair, dir_str, z, dur_str, prob, pnl)

    def update_blotter_table(self):
        table = self.query_one("#table_blotter", DataTable)
        table.clear()
        trades = self.state.get("recent_trades", [])
        if not trades:
            table.add_row("No closed trades", "", "", "", "", "")
            return
            
        for trade in trades:
            pair = trade.get("pair")
            reason = trade.get("exit_reason")
            p1_in = f"{trade.get('entry_p1'):.4f}"
            p1_out = f"{trade.get('exit_p1'):.4f}"
            pnl = trade.get("net_pnl", 0.0)
            pnl_style = GREEN if pnl >= 0 else RED
            pnl_str = Text(f"${pnl:+,.2f}", style=pnl_style)
            
            ret_pct = (pnl / (trade.get("allocated_margin", 1.0))) * 100
            ret_str = Text(f"{ret_pct:+.2f}%", style=pnl_style)
            
            table.add_row(pair, reason, p1_in, p1_out, pnl_str, ret_str)

    def update_radar_grid(self):
        container = self.query_one("#radar_grid")
        # Remove old widgets
        for child in list(container.children):
            child.remove()
        
        active_pairs = {p.get("pair"): p for p in self.state.get("open_positions", [])}
        
        content_lines = []
        for pair in self.state.get("bundle", {}).get("pairs", []):
            if pair in active_pairs:
                p = active_pairs[pair]
                status = "LONG" if p.get("trade_dir") == 1 else "SHORT"
                z = p.get('entry_z', 0.0)
                alpha = p.get('adjusted_alpha', 0.0) * 100
                content_lines.append(f"[{GREEN}]●[/] {pair}: [{YELLOW}]OPEN {status}[/]  Z={z:.2f}  α={alpha:.2f}%")
            else:
                content_lines.append(f"[{GREY}]○[/] {pair}: IDLE")
        
        static = Static("\n".join(content_lines) if content_lines else "No pairs configured")
        container.mount(static)

    def action_switch_tab(self, tab: str):
        self.query_one("#tabs").active = tab

    def action_toggle_pause(self):
        self.paused = not self.paused
        self.query_one("#overlay_paused").styles.display = "block" if self.paused else "none"

    def action_refresh_now(self):
        asyncio.create_task(self.update_state())

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected):
        table = event.control
        if table.id == "table_positions":
            row_idx = event.cursor_row
            positions = self.state.get("open_positions", [])
            if 0 <= row_idx < len(positions):
                self.push_screen(PositionDetailScreen(positions[row_idx]))

if __name__ == "__main__":
    app = LeadQuantDashboard()
    app.run()
