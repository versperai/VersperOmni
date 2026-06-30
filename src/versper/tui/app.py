"""
VersperOmni TUI — Training + Inference terminal UI.

Inspired by the Burn framework TUI (Rust / ratatui).

Usage:
    versper-tui
    python -m versper.tui
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Footer, Header, Input, Label, ListItem, ListView,
    ProgressBar, RichLog, Select, Static, TabbedContent, TabPane,
    TextArea, Checkbox,
)
from textual.message import Message
import torch

from versper.config import MiniMindConfig, VLMConfig, OmniConfig
from versper.model import MiniMindForCausalLM
from versper.vlm import MiniMindVLM
from versper.omni import MiniMindOmni

from versper.tui.widgets import (
    BrailleLineChart,
    ETAEstimator,
    COLOR_TRAIN,
    COLOR_VALID,
    COLOR_TEXT,
    COLOR_DIM,
    COLOR_ACCENT,
    sparkline,
    metric_row,
    status_row,
    format_eta,
)
from rich.style import Style as RichStyle
from rich.text import Text as RichText


# ── Constants ────────────────────────────────────────────────────────────

MODEL_TYPES = {
    "LM (MiniMind)": {
        "config": MiniMindConfig,
        "model": MiniMindForCausalLM,
        "desc": "Text-only language model",
        "extras": {},
    },
    "VLM (MiniMind-V)": {
        "config": VLMConfig,
        "model": MiniMindVLM,
        "desc": "Vision-language model (text + image)",
        "extras": {"vision_model_path": None},
    },
    "Omni (MiniMind-O)": {
        "config": OmniConfig,
        "model": MiniMindOmni,
        "desc": "Full omni model (text + speech + image)",
        "extras": {"audio_encoder_path": None, "vision_model_path": None},
    },
}

DEVICE_CHOICES = ["auto", "cpu", "cuda:0"]

# ── CSS ──────────────────────────────────────────────────────────────────

TUI_CSS = """
Screen {
    background: #1a1a2e;
}

Header {
    background: #16213e;
    color: #e2b714;
    text-style: bold;
}

Footer {
    background: #16213e;
    color: #a0a0b0;
}

TabbedContent {
    height: 1fr;
}

TabPane {
    padding: 1;
}

/* ── Inference Tab ────────────────────────────────────── */

#infer-layout {
    height: 1fr;
}

#infer-config-box {
    height: auto;
    border: solid #0f3460;
    padding: 1;
    margin-bottom: 1;
}

#infer-config-box > Label {
    color: #e2b714;
    text-style: bold;
    margin-bottom: 1;
}

#infer-controls {
    height: auto;
    margin-top: 1;
}

#infer-input-area {
    height: 1fr;
    min-height: 6;
    border: solid #0f3460;
    padding: 1;
}

#infer-output-area {
    height: 2fr;
    min-height: 8;
    border: solid #0f3460;
    padding: 1;
    margin-top: 1;
}

.model-select {
    height: auto;
}

/* ── Training Tab ─────────────────────────────────────── */

#train-layout {
    height: 1fr;
}

#train-config-box {
    height: auto;
    border: solid #0f3460;
    padding: 1;
    margin-bottom: 1;
}

#train-config-box > Label {
    color: #e2b714;
    text-style: bold;
    margin-bottom: 1;
}

/* Left panel: controls + status + text metrics */
#train-left-panel {
    height: 1fr;
    border: solid #0f3460;
    padding: 0;
    margin-right: 1;
}

/* Right panel: loss chart */
#train-right-panel {
    height: 1fr;
    border: solid #0f3460;
    padding: 0;
}

/* Bottom progress bars */
#train-progress-panel {
    height: 5;
    border: solid #0f3460;
    padding: 0 1;
    margin-top: 1;
}

#train-progress-panel > Label {
    color: #e2b714;
    text-style: bold;
}

/* Controls / Status / Metrics sections inside left panel */
.training-section {
    height: auto;
    margin: 0;
}

.training-section Label.title {
    color: #e2b714;
    text-style: bold;
    margin-bottom: 1;
}

/* The split container for left/right */
#train-main-area {
    height: 1fr;
    min-height: 12;
}

/* Progress bar rows */
.progress-row {
    height: 1;
    margin: 0;
}

/* ── Logs Tab ────────────────────────────────────────── */

#log-area {
    height: 1fr;
    border: solid #0f3460;
}

/* ── Shared ──────────────────────────────────────────── */

Label.title {
    color: #e2b714;
    text-style: bold;
    margin-bottom: 1;
}

Button {
    margin-right: 1;
}

Button.-primary {
    background: #0f3460;
    color: #e2b714;
}

Button.-success {
    background: #1a6b3c;
    color: #ffffff;
}

Button.-error {
    background: #6b1a1a;
    color: #ffffff;
}

Select {
    margin-bottom: 1;
}

RichLog {
    background: #0d1b2a;
    color: #a0b0c0;
}

ProgressBar {
    margin-bottom: 0;
}

.metric-row {
    height: 3;
}

.metric-value {
    color: #4fc3f7;
    text-style: bold;
}

#model-status {
    color: #66bb6a;
    text-style: italic;
    margin-top: 1;
}

/* Status text styling */
.status-label {
    color: #e2b714;
}
.status-value {
    color: #a0b0c0;
}

/* Training metrics display */
#train-metrics-text {
    height: 1fr;
    min-height: 4;
    border: none;
}

/* Controls display */
#train-controls-text {
    height: auto;
    border: none;
    margin: 0;
}

/* Status display */
#train-status-text {
    height: auto;
    border: none;
    margin: 0;
}

/* ETA label */
#train-eta-label {
    color: #6b7280;
    text-style: italic;
}
"""


# ── Help Screen ──────────────────────────────────────────────────────────

class HelpScreen(ModalScreen):
    """Keyboard shortcuts help."""

    def compose(self) -> ComposeResult:
        yield Static(
            "\n".join([
                "[bold #e2b714]VersperOmni TUI - Keyboard Shortcuts[/]",
                "",
                "  [bold]q[/]         Quit application",
                "  [bold]h[/]         Show this help screen",
                "  [bold]tab[/]       Switch between tabs",
                "  [bold]ctrl+i[/]    Go to Inference tab",
                "  [bold]ctrl+t[/]    Go to Training tab",
                "  [bold]ctrl+l[/]    Go to Logs tab",
                "",
                "[bold #e2b714]Inference Tab[/]",
                "  [bold]ctrl+g[/]    Generate response",
                "  [bold]ctrl+c[/]    Clear output",
                "",
                "[bold #e2b714]Training Tab[/]",
                "  [bold]ctrl+s[/]    Start/stop training",
                "  [bold]← →[/]       Switch between plotted metrics (TODO)",
                "",
                "Press [bold]Esc[/] or [bold]h[/] to close.",
            ]),
            id="help-text",
        )
        yield Button("Close", variant="primary", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.pop_screen()

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.app.pop_screen()


# ── Quit Confirmation Screen ────────────────────────────────────────────

class QuitScreen(ModalScreen):
    """Quit confirmation popup — inspired by Burn TUI popup."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                "\n".join([
                    "[bold #e2b714]Quit?[/]",
                    "",
                    "  [bold][s][/] Stop the training[/]",
                    "                    Stop the training loop gracefully.",
                    "",
                    "  [bold][k][/] Kill[/]",
                    "                    Kill the training immediately.",
                    "",
                    "  [bold][c][/] Cancel[/]",
                    "                    Cancel and continue.",
                ]),
                id="quit-text",
            ),
            id="quit-dialog",
        )

    def on_key(self, event) -> None:
        app = self.app
        if not isinstance(app, VersperTUI):
            return
        if event.key == "s":
            # Stop gracefully
            if app.training_running:
                app.training_running = False
                if app._train_process:
                    app._train_process.terminate()
                app.post_message(LogMessage("Training stopped by user (graceful)"))
                app.post_message(TrainingDone())
            self.app.pop_screen()
        elif event.key == "k":
            # Kill immediately
            if app.training_running:
                app.training_running = False
                if app._train_process:
                    app._train_process.kill()
                app.post_message(LogMessage("Training killed by user", "error"))
                app.post_message(TrainingDone())
            self.app.pop_screen()
        elif event.key == "c":
            # Cancel
            self.app.pop_screen()
        elif event.key == "escape":
            self.app.pop_screen()


# ── Training Done Screen ────────────────────────────────────────────────

class TrainingDoneScreen(ModalScreen):
    """Post-training popup — inspired by Burn TUI persistent mode."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                "\n".join([
                    "[bold #e2b714]Training Complete[/]",
                    "",
                    "  Press [bold]x[/] to close this dialog.",
                    "  Press [bold]q[/] to quit the application.",
                ]),
                id="done-text",
            ),
            id="done-dialog",
        )

    def on_key(self, event) -> None:
        if event.key == "x":
            self.app.pop_screen()
        elif event.key == "q":
            if isinstance(self.app, VersperTUI):
                self.app.exit()


# ── CSS for popup screens ───────────────────────────────────────────────

POPUP_CSS = """
QuitScreen, TrainingDoneScreen {
    align: center middle;
    background: rgba(0,0,0,0.7);
}

#quit-dialog, #done-dialog {
    width: 50;
    height: auto;
    border: solid #e2b714;
    background: #1a1a2e;
    padding: 1 2;
}

#quit-text, #done-text {
    color: #a0b0c0;
}
"""


# ── Main App ─────────────────────────────────────────────────────────────

class VersperTUI(App):
    """VersperOmni Terminal UI for training and inference."""

    CSS = TUI_CSS + POPUP_CSS
    TITLE = "VersperOmni TUI"

    BINDINGS = [
        Binding("q", "quit_or_popup", "Quit"),
        Binding("h", "show_help", "Help"),
        Binding("ctrl+i", "tab_inference", "Inference"),
        Binding("ctrl+t", "tab_training", "Training"),
        Binding("ctrl+l", "tab_logs", "Logs"),
        Binding("ctrl+g", "generate", "Generate", show=False),
        Binding("ctrl+c", "clear_output", "Clear", show=False),
        Binding("ctrl+s", "toggle_training", "Start/Stop", show=False),
        Binding("left", "prev_metric", "Prev metric", show=False),
        Binding("right", "next_metric", "Next metric", show=False),
        Binding("up", "cycle_plot_kind", "Plot kind", show=False),
        Binding("down", "cycle_plot_kind", "Plot kind", show=False),
    ]

    model_loaded = reactive(False)
    current_model_type = reactive("LM (MiniMind)")
    training_running = reactive(False)

    def __init__(self):
        super().__init__()
        self._model = None
        self._config = None
        self._tokenizer = None
        self._train_process: Optional[subprocess.Popen] = None
        self._train_thread: Optional[threading.Thread] = None
        self._log_lines: list[str] = []

        # Training metric tracking
        self._loss_history: list[float] = []
        self._lr_history: list[float] = []
        self._epoch = 0
        self._total_epochs = 0
        self._step = 0
        self._total_steps = 0
        self._current_lr = 0.0
        self._current_loss = 0.0
        self._eta_estimator = ETAEstimator()
        self._train_start_time: Optional[float] = None

        # Last progress bar values (for rendering in custom widgets)
        self._epoch_progress = 0.0
        self._batch_progress = 0.0

        # Burn-inspired plot selection state
        self._plot_kinds = ["loss", "lr"]
        self._plot_selected = 0  # index into _plot_kinds
        self._available_kinds = ["Full", "Recent", "Summary"]
        self._plot_kind_idx = 0  # 0=Full, 1=Recent, 2=Summary

    # ── Compose ──────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-inference"):
            with TabPane("🖥  Inference", id="tab-inference"):
                yield self._build_inference_tab()
            with TabPane("🎯  Training", id="tab-training"):
                yield self._build_training_tab()
            with TabPane("📋  Logs", id="tab-logs"):
                yield self._build_logs_tab()
        yield Footer()

    def _build_inference_tab(self) -> Container:
        return Container(
            Vertical(
                # Config section
                Container(
                    Label("Model Configuration", classes="title"),
                    Horizontal(
                        Vertical(
                            Label("Model Type"),
                            Select(
                                [(k, k) for k in MODEL_TYPES],
                                prompt="Select model...",
                                id="infer-model-type",
                                value="LM (MiniMind)",
                            ),
                        ),
                        Vertical(
                            Label("Device"),
                            Select(
                                [(d, d) for d in DEVICE_CHOICES],
                                prompt="Select device...",
                                id="infer-device",
                                value="auto",
                            ),
                        ),
                        classes="model-select",
                    ),
                    Horizontal(
                        Label("Weight path:"),
                        Input(placeholder="./out/model.pth", id="infer-weight-path"),
                    ),
                    Horizontal(
                        Button("Load Model", variant="primary", id="btn-load-model"),
                        Button("Unload", variant="error", id="btn-unload-model"),
                        id="infer-controls",
                    ),
                    Static("No model loaded", id="model-status"),
                    id="infer-config-box",
                ),
                # Input area
                Container(
                    Label("Input", classes="title"),
                    TextArea(id="infer-input", text="Type your prompt here..."),
                    Horizontal(
                        Button("Generate (Ctrl+G)", variant="success", id="btn-generate"),
                        Button("Clear (Ctrl+C)", id="btn-clear"),
                        Checkbox("Stream output", id="infer-stream", value=False),
                    ),
                    id="infer-input-area",
                ),
                # Output area
                Container(
                    Label("Output", classes="title"),
                    RichLog(id="infer-output", highlight=True, markup=True, wrap=True),
                    id="infer-output-area",
                ),
                id="infer-layout",
            ),
        )

    def _build_training_tab(self) -> Container:
        return Container(
            Vertical(
                # Config section (collapsible-like at top)
                self._train_config_section(),
                # Main split area: left panel + right panel
                Horizontal(
                    # Left panel: Controls + Status + Text Metrics
                    ScrollableContainer(
                        Static(id="train-controls-text"),
                        Static(id="train-status-text"),
                        RichLog(
                            id="train-metrics-text",
                            highlight=True,
                            markup=True,
                            wrap=True,
                        ),
                        id="train-left-panel",
                    ),
                    # Right panel: Loss chart
                    ScrollableContainer(
                        BrailleLineChart(
                            id="train-loss-chart",
                            title="Training Loss",
                            height_chars=8,
                            color="#e2b714",
                        ),
                        id="train-right-panel",
                    ),
                    id="train-main-area",
                ),
                # Bottom: dual progress bars with ETA
                Container(
                    Label("Progress", classes="title"),
                    Horizontal(
                        Static("Epoch:", classes="progress-row"),
                        ProgressBar(total=100, id="train-epoch-progress", show_eta=False),
                        Static(id="train-eta-label"),
                    ),
                    Horizontal(
                        Static("Batch:", classes="progress-row"),
                        ProgressBar(total=100, id="train-batch-progress", show_eta=False),
                    ),
                    id="train-progress-panel",
                ),
                id="train-layout",
            ),
        )

    def _train_config_section(self) -> Container:
        return Container(
            Label("Training Configuration", classes="title"),
            Horizontal(
                Vertical(
                    Label("Training Mode"),
                    Select(
                        [
                            ("Pretrain (LM)", "pretrain"),
                            ("SFT-VLM", "sft_vlm"),
                            ("SFT-Omni", "sft_omni"),
                        ],
                        prompt="Select mode...",
                        id="train-mode",
                        value="pretrain",
                    ),
                ),
                Vertical(
                    Label("Data path"),
                    Input(placeholder="../dataset/data.jsonl", id="train-data-path"),
                ),
                Vertical(
                    Label("Model type"),
                    Select(
                        [(k, k) for k in MODEL_TYPES],
                        prompt="Select...",
                        id="train-model-type",
                        value="LM (MiniMind)",
                    ),
                ),
            ),
            Horizontal(
                Label("Weight path:"),
                Input(placeholder="./out/pretrain.pth", id="train-weight-path"),
            ),
            Horizontal(
                Label("Device:"),
                Select(
                    [(d, d) for d in DEVICE_CHOICES],
                    id="train-device",
                    value="auto",
                ),
                Label("Batch size:"),
                Input(value="32", id="train-batch-size", type="integer"),
                Label("Learning rate:"),
                Input(value="5e-4", id="train-lr"),
            ),
            Horizontal(
                Button("▶ Start Training", variant="success", id="btn-train-start"),
                Button("⏹ Stop", variant="error", id="btn-train-stop"),
                Button("Clear Metrics", id="btn-train-clear"),
            ),
            id="train-config-box",
        )

    def _build_logs_tab(self) -> Container:
        return Container(
            Vertical(
                Horizontal(
                    Button("Clear Logs", id="btn-logs-clear"),
                    Button("Copy Logs", id="btn-logs-copy"),
                ),
                RichLog(id="log-area", highlight=True, markup=True, wrap=True, max_lines=10000),
            ),
        )

    # ── Actions ──────────────────────────────────────────────────────

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_quit_or_popup(self) -> None:
        """If training is running, show quit confirmation; otherwise exit."""
        if self.training_running:
            self.push_screen(QuitScreen())
        else:
            self.exit()

    def action_tab_inference(self) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = "tab-inference"
        except NoMatches:
            pass

    def action_tab_training(self) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = "tab-training"
        except NoMatches:
            pass

    def action_tab_logs(self) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = "tab-logs"
        except NoMatches:
            pass

    def action_generate(self) -> None:
        """Ctrl+G shortcut for generate in inference tab."""
        try:
            prompt = self.query_one("#infer-input").text
            stream = self.query_one("#infer-stream").value
            self.run_inference(prompt, stream)
        except NoMatches:
            pass

    def action_clear_output(self) -> None:
        """Ctrl+C shortcut for clearing inference output."""
        try:
            self.query_one("#infer-output").clear()
            self.query_one("#infer-input").text = ""
        except NoMatches:
            pass

    def action_prev_metric(self) -> None:
        """←: switch to previous plot metric."""
        if self._plot_kinds:
            self._plot_selected = (self._plot_selected - 1) % len(self._plot_kinds)
            self._refresh_chart_metric()

    def action_next_metric(self) -> None:
        """→: switch to next plot metric."""
        if self._plot_kinds:
            self._plot_selected = (self._plot_selected + 1) % len(self._plot_kinds)
            self._refresh_chart_metric()

    def action_cycle_plot_kind(self) -> None:
        """↑/↓: cycle through Full / Recent / Summary plot kinds."""
        self._plot_kind_idx = (self._plot_kind_idx + 1) % len(self._available_kinds)
        self._refresh_chart_title()

    def _refresh_chart_metric(self) -> None:
        """Rebuild chart with currently selected metric."""
        try:
            chart = self.query_one("#train-loss-chart")
            if not isinstance(chart, BrailleLineChart):
                return
            chart.clear()
            metric = self._plot_kinds[self._plot_selected]
            data = getattr(self, f"_{metric}_history", [])
            chart.add_series(metric, color="#e2b714")
            for i, val in enumerate(data):
                chart.add_point(metric, i, val)
            # Update status
            try:
                status = self.query_one("#train-status-text")
                kind = self._available_kinds[self._plot_kind_idx]
                t = RichText()
                t.append(" Status", style=RichStyle(color="#e2b714", bold=True))
                t.append(f"\n  Plot: ", style=RichStyle(color="#6b7280"))
                t.append(f"{metric}", style=RichStyle(color="#e2b714", bold=True))
                t.append(f" [{kind}]", style=RichStyle(color="#4fc3f7"))
                if self._current_loss:
                    t.append(f"\n  Loss: ", style=RichStyle(color="#6b7280"))
                    t.append(f"{self._current_loss:.4f}", style=RichStyle(color="#a0b0c0"))
                if self._current_lr:
                    t.append(f"\n  LR:   ", style=RichStyle(color="#6b7280"))
                    t.append(f"{self._current_lr:.6f}", style=RichStyle(color="#a0b0c0"))
                t.append(f"\n  Epoch: ", style=RichStyle(color="#6b7280"))
                t.append(f"{self._epoch}/{self._total_epochs}", style=RichStyle(color="#a0b0c0"))
                if self._step and self._total_steps:
                    t.append(f"\n  Step: ", style=RichStyle(color="#6b7280"))
                    t.append(f"{self._step}/{self._total_steps}", style=RichStyle(color="#a0b0c0"))
                status.update(t)
            except NoMatches:
                pass
        except NoMatches:
            pass

    def _refresh_chart_title(self) -> None:
        """Update chart title to reflect selected metric and plot kind."""
        try:
            chart = self.query_one("#train-loss-chart")
            if not isinstance(chart, BrailleLineChart):
                return
            metric = self._plot_kinds[self._plot_selected]
            kind = self._available_kinds[self._plot_kind_idx]
            chart._title = f"{metric} ({kind})"
            chart.refresh()
        except NoMatches:
            pass

    def action_toggle_training(self) -> None:
        """Ctrl+S shortcut for start/stop training."""
        try:
            if self.training_running:
                self.training_running = False
                if self._train_process:
                    self._train_process.kill()
                self.post_message(LogMessage("Training stopped by shortcut"))
            else:
                # Find and press start button
                btn = self.query_one("#btn-train-start")
                self._trigger_train_start()
        except NoMatches:
            pass

    # ── Model Loading ────────────────────────────────────────────────

    def _get_device(self, device_key: str) -> str:
        if device_key == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device_key

    @work(thread=True)
    def load_model(self, model_key: str, device_key: str, weight_path: str) -> None:
        """Load model in background thread."""
        try:
            self.post_message(LogMessage(f"Loading {model_key} model..."))
            model_info = MODEL_TYPES[model_key]
            config_cls = model_info["config"]
            model_cls = model_info["model"]
            device = self._get_device(device_key)

            cfg = config_cls()
            self._config = cfg

            extras = model_info["extras"].copy()
            self._model = model_cls(cfg, **extras)

            if weight_path and os.path.exists(weight_path):
                state = torch.load(weight_path, map_location="cpu")
                self._model.load_state_dict(state, strict=False)
                self.post_message(LogMessage(f"Loaded weights from {weight_path}"))
            else:
                self.post_message(LogMessage("No weights loaded (random init)", "warning"))

            self._model = self._model.to(device).eval()
            self.current_model_type = model_key
            self.model_loaded = True

            self.post_message(ModelLoaded(model_key, device))
        except Exception as e:
            self.post_message(LogMessage(f"Failed to load model: {e}", "error"))
            self.model_loaded = False

    def on_model_loaded(self) -> None:
        """Reactive: update UI when model loads."""
        try:
            status = self.query_one("#model-status")
            if self.model_loaded:
                dev = self._get_device(
                    self.query_one("#infer-device").value
                )
                status.update(f"✅ {self.current_model_type} loaded on {dev}")
                status.styles.color = "#66bb6a"
            else:
                status.update("❌ Failed to load model")
                status.styles.color = "#ef5350"
        except NoMatches:
            pass

    # ── Inference ────────────────────────────────────────────────────

    @work(thread=True)
    def run_inference(self, prompt: str, stream: bool) -> None:
        """Run model inference in background thread."""
        if self._model is None:
            self.post_message(LogMessage("No model loaded!", "error"))
            self.post_message(InferenceResult("No model loaded. Please load a model first."))
            return

        try:
            self.post_message(LogMessage(f"Inferring: {prompt[:60]}..."))
            model = self._model

            if self.current_model_type == "LM (MiniMind)":
                from transformers import AutoTokenizer
                tok_path = self.query_one("#infer-weight-path").value or "./model"
                if self._tokenizer is None:
                    if os.path.exists(tok_path):
                        self._tokenizer = AutoTokenizer.from_pretrained(tok_path)
                    else:
                        self.post_message(InferenceResult(
                            "Tokenizer not found at ./model. Please download tokenizer files."
                        ))
                        return

                messages = [{"role": "user", "content": prompt}]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = self._tokenizer(text, return_tensors="pt").input_ids
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)

                with torch.inference_mode():
                    out = model.generate(
                        input_ids, max_new_tokens=512,
                        temperature=0.85, top_p=0.85,
                    )
                response = self._tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True
                )
                self.post_message(InferenceResult(response))

            elif self.current_model_type == "VLM (MiniMind-V)":
                img_match = re.search(r'!\[.*?\]\((.+?)\)', prompt)
                if img_match:
                    from PIL import Image
                    img_path = img_match.group(1)
                    img = Image.open(img_path)
                    text_prompt = re.sub(r'!\[.*?\]\(.+?\)', '', prompt).strip()
                    if not text_prompt:
                        text_prompt = "Describe this image"
                    inputs = model.processor(
                        text=text_prompt, images=img, return_tensors="pt"
                    )
                    device = next(model.parameters()).device
                    input_ids = inputs["input_ids"].to(device)
                    pixel_values = inputs["pixel_values"].to(device)
                    with torch.inference_mode():
                        out = model.generate(
                            input_ids, pixel_values=pixel_values,
                            max_new_tokens=256,
                        )
                    response = model.processor.tokenizer.decode(
                        out[0][input_ids.shape[1]:], skip_special_tokens=True
                    )
                else:
                    inputs = model.processor(text=prompt, return_tensors="pt")
                    device = next(model.parameters()).device
                    input_ids = inputs["input_ids"].to(device)
                    with torch.inference_mode():
                        out = model.generate(input_ids, max_new_tokens=256)
                    response = model.processor.tokenizer.decode(
                        out[0][input_ids.shape[1]:], skip_special_tokens=True
                    )
                    response = "(No image provided)\n\n" + response
                self.post_message(InferenceResult(response))

            elif self.current_model_type == "Omni (MiniMind-O)":
                from transformers import AutoTokenizer
                tok_path = self.query_one("#infer-weight-path").value or "./model"
                if self._tokenizer is None:
                    if os.path.exists(tok_path):
                        self._tokenizer = AutoTokenizer.from_pretrained(tok_path)
                    else:
                        self.post_message(InferenceResult(
                            "Tokenizer not found. Text generation only."
                        ))
                        input_ids = torch.randint(0, 100, (1, 8))
                        device = next(model.parameters()).device
                        input_ids = input_ids.to(device)
                        self._tokenizer = None

                if self._tokenizer is not None:
                    messages = [{"role": "user", "content": prompt}]
                    text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    input_ids = self._tokenizer(text, return_tensors="pt").input_ids
                else:
                    input_ids = torch.randint(0, 100, (1, 8))

                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                with torch.inference_mode():
                    out = model.generate(input_ids, max_new_tokens=256, stream=False)

                if isinstance(out, tuple):
                    tokens, audio = out
                    if self._tokenizer is not None:
                        response = self._tokenizer.decode(
                            tokens[0] if tokens.dim() > 1 else tokens,
                            skip_special_tokens=True,
                        )
                    else:
                        response = f"[Omni output: audio codes shape {audio.shape}]"
                else:
                    response = str(out)

                self.post_message(InferenceResult(response))

            self.post_message(LogMessage("Inference complete"))
        except Exception as e:
            self.post_message(LogMessage(f"Inference error: {e}", "error"))
            self.post_message(InferenceResult(f"Error: {e}"))

    # ── Training ─────────────────────────────────────────────────────

    @work(thread=True)
    def run_training(self, mode: str, data_path: str, model_type: str,
                     weight_path: str, device: str, batch_size: int, lr: float) -> None:
        """Launch training as subprocess and monitor output."""
        self.training_running = True
        self._train_start_time = time.time()
        self._eta_estimator.start()

        # Reset metric tracking
        self._loss_history.clear()
        self._lr_history.clear()
        self._epoch = 0
        self._total_epochs = 0
        self._step = 0
        self._total_steps = 0

        try:
            cmd = [
                sys.executable, "-m", f"versper.trainer.{mode}",
                "--data_path", data_path,
                "--batch_size", str(batch_size),
                "--learning_rate", str(lr),
                "--device", self._get_device(device),
            ]
            if weight_path and os.path.exists(weight_path):
                cmd.extend(["--from_weight", weight_path.replace(".pth", "")])

            self.post_message(LogMessage(f"Starting: {' '.join(cmd)}"))

            self._train_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Read output line by line
            for line in iter(self._train_process.stdout.readline, ""):
                if not self.training_running:
                    self._train_process.kill()
                    break

                line = line.strip()
                if not line:
                    continue

                self.post_message(LogMessage(line, "train"))

                # Parse metrics from log lines
                if "loss:" in line and "Epoch:" in line:
                    self._parse_train_metrics(line)

            self._train_process.wait()
            self.post_message(LogMessage(f"Training exited with code {self._train_process.returncode}"))
            self.post_message(TrainingDone())

        except Exception as e:
            self.post_message(LogMessage(f"Training error: {e}", "error"))
        finally:
            self.training_running = False
            self._train_process = None

    def _parse_train_metrics(self, line: str) -> None:
        """Parse training metric values from a log line."""
        try:
            ep = re.search(r'Epoch:\[(\d+)/(\d+)\]', line)
            step = re.search(r'\((\d+)/(\d+)\)', line)
            loss = re.search(r'loss: ([\d.]+)', line)
            lr_m = re.search(r'lr: ([\d.]+)', line)

            if ep:
                self._epoch = int(ep.group(1))
                self._total_epochs = int(ep.group(2))
                self.post_message(TrainProgress(
                    epoch=self._epoch,
                    total_epochs=self._total_epochs,
                ))

            if step and loss:
                step_num = int(step.group(1))
                total_steps = int(step.group(2))
                loss_val = float(loss.group(1))
                lr_val = float(lr_m.group(1)) if lr_m else 0.0

                self._step = step_num
                self._total_steps = total_steps
                self._current_loss = loss_val
                self._current_lr = lr_val
                self._loss_history.append(loss_val)
                self._lr_history.append(lr_val)

                # Cumulative step across epochs
                cum_step = (self._epoch - 1) * total_steps + step_num
                self._eta_estimator.update(cum_step)

                self.post_message(TrainMetric(
                    loss=loss_val,
                    step=step_num,
                    total_steps=total_steps,
                    lr=lr_val,
                    epoch=self._epoch,
                    total_epochs=self._total_epochs,
                ))

                # Update loss chart with currently selected metric
                metric = self._plot_kinds[self._plot_selected]
                if metric == "loss":
                    self.post_message(LossPoint(loss_val, lr_val))
                elif metric == "lr":
                    self.post_message(LossPoint(lr_val, loss_val))  # swap: x=step, y=lr
        except (ValueError, AttributeError):
            pass

    def _trigger_train_start(self) -> None:
        """Collect training config and launch."""
        mode = self.query_one("#train-mode").value
        data_path = self.query_one("#train-data-path").value.strip()
        model_type = self.query_one("#train-model-type").value
        weight_path = self.query_one("#train-weight-path").value.strip()
        device = self.query_one("#train-device").value
        try:
            batch_size = int(self.query_one("#train-batch-size").value)
        except ValueError:
            batch_size = 32
        try:
            lr = float(self.query_one("#train-lr").value)
        except ValueError:
            lr = 5e-4

        self.run_training(mode, data_path, model_type, weight_path, device, batch_size, lr)

    # ── Message Handling ─────────────────────────────────────────────

    def on_log_message(self, msg: "LogMessage") -> None:
        """Handle log messages."""
        prefix = {
            "warning": "[yellow]⚠[/]",
            "error": "[red]✗[/]",
            "train": "[cyan]▶[/]",
        }.get(msg.level, "[dim]•[/]")

        log_line = f"{prefix} {msg.text}"
        self._log_lines.append(log_line)

        try:
            log_area = self.query_one("#log-area")
            log_area.write(log_line + "\n")
        except NoMatches:
            pass

        # Write to train metrics text area as well
        if msg.level == "train":
            try:
                tm = self.query_one("#train-metrics-text")
                tm.write(log_line + "\n")
            except NoMatches:
                pass

    def on_inference_result(self, msg: "InferenceResult") -> None:
        """Handle inference results."""
        try:
            output = self.query_one("#infer-output")
            output.clear()
            output.write(f"\n[bold #e2b714]Response:[/]\n{msg.text}\n")
        except NoMatches:
            pass

    def on_train_metric(self, msg: "TrainMetric") -> None:
        """Handle training metrics update — progress bars + status panel."""
        try:
            # Update progress bars
            epoch_bar = self.query_one("#train-epoch-progress")
            batch_bar = self.query_one("#train-batch-progress")

            if msg.total_epochs:
                epoch_progress = (msg.epoch / msg.total_epochs) * 100
                epoch_bar.update(progress=epoch_progress)
                self._epoch_progress = epoch_progress

            if msg.total_steps:
                batch_progress = (msg.step / msg.total_steps) * 100
                batch_bar.update(progress=batch_progress)
                self._batch_progress = batch_progress

            # Update status panel
            self._update_status_panel(msg)

            # Update ETA label
            try:
                eta = self.query_one("#train-eta-label")
                remaining_epochs = msg.total_epochs - msg.epoch
                remaining_steps = remaining_epochs * msg.total_steps + (msg.total_steps - msg.step)
                eta_str = self._eta_estimator.eta_str_for(remaining_steps)
                eta.update(f" ETA: {eta_str}")
            except NoMatches:
                pass

        except NoMatches:
            pass

    def _update_status_panel(self, msg: "TrainMetric") -> None:
        """Refresh the status/controls/metrics text in the left panel."""
        try:
            status = self.query_one("#train-status-text")
            t = RichText()
            t.append(" Status", style=RichStyle(color="#e2b714", bold=True))
            status.update(t)
        except NoMatches:
            pass

    def on_train_progress(self, msg: "TrainProgress") -> None:
        """Handle epoch progress updates."""
        try:
            epoch_bar = self.query_one("#train-epoch-progress")
            if msg.total_epochs:
                pct = (msg.epoch / msg.total_epochs) * 100
                epoch_bar.update(progress=pct)
                self._epoch_progress = pct
        except NoMatches:
            pass

    def on_loss_point(self, msg: "LossPoint") -> None:
        """Handle new data point — update chart and status."""
        try:
            chart = self.query_one("#train-loss-chart")
            if isinstance(chart, BrailleLineChart):
                metric = self._plot_kinds[self._plot_selected]
                chart.add_series(metric, color="#e2b714")
                if metric == "loss":
                    chart.add_point(metric, len(self._loss_history), msg.loss)
                else:
                    chart.add_point(metric, len(self._lr_history), msg.lr)
                chart.refresh()
        except NoMatches:
            pass

        # Update status panel with latest values
        try:
            status = self.query_one("#train-status-text")
            metric = self._plot_kinds[self._plot_selected]
            kind = self._available_kinds[self._plot_kind_idx]
            t = RichText()
            t.append(" Status", style=RichStyle(color="#e2b714", bold=True))
            t.append(f"\n  Plot: ", style=RichStyle(color="#6b7280"))
            t.append(f"{metric}", style=RichStyle(color="#e2b714", bold=True))
            t.append(f" [{kind}]", style=RichStyle(color="#4fc3f7"))
            if self._current_loss:
                t.append(f"\n  Loss: ", style=RichStyle(color="#6b7280"))
                t.append(f"{self._current_loss:.4f}", style=RichStyle(color="#a0b0c0"))
            if self._current_lr:
                t.append(f"\n  LR:   ", style=RichStyle(color="#6b7280"))
                t.append(f"{self._current_lr:.6f}", style=RichStyle(color="#a0b0c0"))
            t.append(f"\n  Epoch: ", style=RichStyle(color="#6b7280"))
            t.append(f"{self._epoch}/{self._total_epochs}", style=RichStyle(color="#a0b0c0"))
            if self._step and self._total_steps:
                t.append(f"\n  Step: ", style=RichStyle(color="#6b7280"))
                t.append(f"{self._step}/{self._total_steps}", style=RichStyle(color="#a0b0c0"))
            # Event counters (burn-inspired)
            if self._loss_history:
                t.append(f"\n  Points: ", style=RichStyle(color="#6b7280"))
                t.append(f"{len(self._loss_history)}", style=RichStyle(color="#a0b0c0"))
            status.update(t)
        except NoMatches:
            pass

    def on_training_done(self, msg: "TrainingDone") -> None:
        """Handle training completion — show popup."""
        self.push_screen(TrainingDoneScreen())

    # ── Button Handlers ──────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        # ── Inference ──
        if button_id == "btn-load-model":
            model_key = self.query_one("#infer-model-type").value
            device_key = self.query_one("#infer-device").value
            weight_path = self.query_one("#infer-weight-path").value.strip()
            self.load_model(model_key, device_key, weight_path)

        elif button_id == "btn-unload-model":
            self._model = None
            self._tokenizer = None
            self.model_loaded = False
            try:
                self.query_one("#model-status").update("Model unloaded")
                self.query_one("#infer-output").clear()
            except NoMatches:
                pass

        elif button_id == "btn-generate":
            prompt = self.query_one("#infer-input").text
            stream = self.query_one("#infer-stream").value
            self.run_inference(prompt, stream)

        elif button_id == "btn-clear":
            try:
                self.query_one("#infer-output").clear()
                self.query_one("#infer-input").text = ""
            except NoMatches:
                pass

        # ── Training ──
        elif button_id == "btn-train-start":
            if self.training_running:
                self.post_message(LogMessage("Training already running!", "warning"))
                return
            self._trigger_train_start()

        elif button_id == "btn-train-stop":
            self.training_running = False
            if self._train_process:
                self._train_process.kill()
            self.post_message(LogMessage("Training stopped by user"))
            self.post_message(TrainingDone())

        elif button_id == "btn-train-clear":
            try:
                self.query_one("#train-metrics-text").clear()
                self._loss_history.clear()
                self._lr_history.clear()
                chart = self.query_one("#train-loss-chart")
                if isinstance(chart, BrailleLineChart):
                    chart.clear()
            except NoMatches:
                pass

        # ── Logs ──
        elif button_id == "btn-logs-clear":
            try:
                self.query_one("#log-area").clear()
            except NoMatches:
                pass

        elif button_id == "btn-logs-copy":
            try:
                import pyperclip
                text = "\n".join(self._log_lines)
                pyperclip.copy(text)
                self.post_message(LogMessage("Logs copied to clipboard"))
            except ImportError:
                self.post_message(LogMessage("pyperclip not available; install with: pip install pyperclip", "warning"))

    # ── Lifecycle ────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.post_message(LogMessage("VersperOmni TUI started"))
        self.post_message(LogMessage("Press 'h' for keyboard shortcuts"))


# ── Custom Messages ──────────────────────────────────────────────────────

class LogMessage(Message):
    def __init__(self, text: str, level: str = "info") -> None:
        super().__init__()
        self.text = text
        self.level = level


class ModelLoaded(Message):
    def __init__(self, model_type: str, device: str) -> None:
        super().__init__()
        self.model_type = model_type
        self.device = device


class InferenceResult(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class TrainMetric(Message):
    def __init__(self, loss: float, step: int, total_steps: int,
                 lr: float, epoch: int = 0, total_epochs: int = 0) -> None:
        super().__init__()
        self.loss = loss
        self.step = step
        self.total_steps = total_steps
        self.lr = lr
        self.epoch = epoch
        self.total_epochs = total_epochs


class TrainProgress(Message):
    def __init__(self, epoch: int, total_epochs: int) -> None:
        super().__init__()
        self.epoch = epoch
        self.total_epochs = total_epochs


class LossPoint(Message):
    def __init__(self, loss: float, lr: float) -> None:
        super().__init__()
        self.loss = loss
        self.lr = lr


class TrainingDone(Message):
    """Sent when training finishes (success, cancel, or kill)."""
    pass


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    app = VersperTUI()
    app.run()


if __name__ == "__main__":
    main()
