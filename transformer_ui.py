"""
Transformer Optimizer UI - Dear PyGui 2.11 Compatible Version
Modern, high-resolution interface with GPU acceleration
"""

import dearpygui.dearpygui as dpg
import threading
import math
import time
import sys
import os
import io
from queue import Queue, Empty

# Cross-platform sound playback
if sys.platform == 'win32':
    import winsound
    def play_sound(path):
        try:
            if os.path.exists(path):
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:
            pass  # Silently ignore sound errors
else:
    try:
        from playsound3 import playsound
        def play_sound(path):
            try:
                playsound(path, block=False)
            except Exception:
                pass  # Silently ignore sound errors
    except ImportError:
        # playsound3 not available, create no-op function
        def play_sound(path):
            pass

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

import mainRect


class ConsoleCapture:
    """Captures stdout/stderr and stores it for later display.

    Uses a thread-safe queue to pass output to the UI thread.
    Also writes to original stdout/stderr so console stays in sync.
    """
    def __init__(self, output_queue=None):
        self.buffer = io.StringIO()
        self.output_queue = output_queue
        self._original_stdout = None
        self._original_stderr = None

    def write(self, text):
        self.buffer.write(text)
        # Always write to original stdout so console stays in sync
        if self._original_stdout:
            try:
                self._original_stdout.write(text)
                self._original_stdout.flush()
            except:
                pass
        # Queue for thread-safe UI update
        if self.output_queue is not None:
            try:
                self.output_queue.put_nowait(text)
            except:
                pass  # Queue full, skip this text

    def flush(self):
        if self._original_stdout:
            try:
                self._original_stdout.flush()
            except:
                pass

    def get_output(self):
        return self.buffer.getvalue()

    def clear(self):
        self.buffer = io.StringIO()

    def start_capture(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def stop_capture(self):
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr


class TransformerOptimizerApp:
    def __init__(self):
        self.is_optimizing = False
        self.optimization_thread = None
        self.result = None
        self.duck_angle = 0
        self._last_duck_update = 0
        self.console_capture = None
        self.console_queue = Queue(maxsize=1000)  # Thread-safe queue for console output

        dpg.create_context()
        dpg.create_viewport(title="Transformer Design Optimizer", width=1100, height=850)

        self._setup_fonts()
        self._setup_theme()
        self._create_ui()

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Register frame callback for thread-safe duck animation
        dpg.set_frame_callback(1, self._on_frame)

    def _setup_fonts(self):
        """Setup fonts."""
        self.default_font = None

    def _setup_theme(self):
        """Setup dark theme."""
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 35))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (40, 40, 48))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 50, 60))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (60, 60, 72))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (70, 70, 85))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 130, 220))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 150, 240))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (60, 110, 200))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 55, 70))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (60, 68, 88))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 80, 100))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 8)

        dpg.bind_theme(self.global_theme)

        # Theme for run button (green)
        with dpg.theme() as self.run_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 150, 90))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 175, 110))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 130, 75))

        # Theme for stop button (red)
        with dpg.theme() as self.stop_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (200, 80, 80))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 45, 45))

        # Theme for section headers
        with dpg.theme() as self.section_theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 55, 70))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (60, 68, 88))

        # Theme for progress bar
        with dpg.theme() as self.progress_theme:
            with dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (70, 130, 220))

    def _create_ui(self):
        """Create the main UI layout."""
        with dpg.window(label="Transformer Optimizer", tag="main_window", no_title_bar=True,
                       no_move=True, no_resize=True, no_collapse=True):

            # Header section with title and duck
            with dpg.group(horizontal=True):
                dpg.add_text("TRANSFORMER DESIGN OPTIMIZER", color=(100, 165, 255))
                dpg.add_spacer(width=20)
                with dpg.drawlist(width=60, height=45, tag="duck_canvas"):
                    self._draw_duck(30, 22, 0)
                dpg.add_spacer(width=20)
                dpg.add_text("GPU-Accelerated", color=(120, 125, 145))

            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Tab bar for main content
            with dpg.tab_bar(tag="main_tabs"):
                # Main Optimization tab
                with dpg.tab(label=" Optimization "):
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        # Left column - Inputs
                        with dpg.child_window(width=440, height=580, border=True, tag="input_panel"):
                            self._create_input_panels()

                        dpg.add_spacer(width=8)

                        # Right column - Results
                        with dpg.child_window(width=440, height=580, border=True, tag="results_panel"):
                            self._create_results_panel()

                # Advanced Materials tab
                with dpg.tab(label=" Material Properties "):
                    dpg.add_spacer(height=4)
                    self._create_materials_panel()

                # Console Output tab
                with dpg.tab(label=" Console Output "):
                    dpg.add_spacer(height=4)
                    self._create_console_panel()

            dpg.add_spacer(height=8)

            # Bottom action bar
            with dpg.group(horizontal=True):
                dpg.add_button(label=" Run Optimization ", tag="run_btn",
                              callback=self._run_optimization, width=160, height=32)
                dpg.bind_item_theme("run_btn", self.run_button_theme)

                dpg.add_spacer(width=8)

                dpg.add_button(label=" Stop ", tag="stop_btn",
                              callback=self._stop_optimization, width=80, height=32, enabled=False)
                dpg.bind_item_theme("stop_btn", self.stop_button_theme)

                dpg.add_spacer(width=30)

                # Status display - using horizontal group instead of deprecated add_same_line
                with dpg.group(horizontal=True):
                    dpg.add_text("Status:", color=(100, 105, 125))
                    dpg.add_text("Ready", tag="status_text", color=(72, 199, 142))

        dpg.set_primary_window("main_window", True)

    def _draw_duck(self, cx, cy, angle):
        """Draw a cute rubber duck at position (cx, cy) with rotation angle."""
        dpg.delete_item("duck_canvas", children_only=True)

        # Wobble effect based on angle
        wobble = math.sin(angle * 4) * 2
        bob = math.sin(angle * 2) * 1.5

        # Colors
        yellow = (255, 215, 50, 255)
        orange = (255, 140, 50, 255)
        black = (25, 25, 30, 255)
        white = (255, 255, 255, 255)
        blue = (100, 150, 220, 80)

        # Water ripple
        ripple_offset = math.sin(angle * 3) * 2
        dpg.draw_ellipse((cx - 18 + ripple_offset, cy + 10), (cx + 18 - ripple_offset, cy + 16),
                        color=blue, fill=blue, parent="duck_canvas")

        # Body (ellipse)
        body_cx = cx + wobble
        body_cy = cy + bob
        dpg.draw_ellipse((body_cx - 14, body_cy - 5), (body_cx + 10, body_cy + 10),
                        color=yellow, fill=yellow, parent="duck_canvas")

        # Head (circle)
        head_cx = body_cx + 8
        head_cy = body_cy - 8 + bob
        dpg.draw_circle((head_cx, head_cy), 8, color=yellow, fill=yellow, parent="duck_canvas")

        # Beak
        beak_points = [
            (head_cx + 6, head_cy - 1),
            (head_cx + 14, head_cy + 1),
            (head_cx + 6, head_cy + 3)
        ]
        dpg.draw_triangle(beak_points[0], beak_points[1], beak_points[2],
                         color=orange, fill=orange, parent="duck_canvas")

        # Eye
        dpg.draw_circle((head_cx + 3, head_cy - 2), 2, color=black, fill=black, parent="duck_canvas")
        dpg.draw_circle((head_cx + 2, head_cy - 3), 0.8, color=white, fill=white, parent="duck_canvas")

        # Wing hint
        dpg.draw_ellipse((body_cx - 5, body_cy), (body_cx + 4, body_cy + 5),
                        color=(240, 195, 40, 255), fill=(240, 195, 40, 255), parent="duck_canvas")

        # Tail feather
        tail_x = body_cx - 13
        tail_y = body_cy - 2 + wobble * 0.5
        dpg.draw_triangle(
            (tail_x, tail_y),
            (tail_x - 5, tail_y - 4),
            (tail_x - 3, tail_y + 2),
            color=yellow, fill=yellow, parent="duck_canvas"
        )

    def _on_frame(self):
        """Frame callback for thread-safe duck animation and console updates (runs in main thread)."""
        # Re-register for next frame
        dpg.set_frame_callback(dpg.get_frame_count() + 1, self._on_frame)

        # Process console queue (thread-safe UI updates)
        self._process_console_queue()

        # Animate duck only when optimizing
        if self.is_optimizing:
            current_time = time.time()
            # Update at ~20 FPS for smooth animation
            if current_time - self._last_duck_update > 0.05:
                self.duck_angle += 0.15
                self._draw_duck(30, 22, self.duck_angle)
                self._last_duck_update = current_time
        elif self.duck_angle != 0:
            # Reset duck to static when optimization ends
            self.duck_angle = 0
            self._draw_duck(30, 22, 0)

    def _process_console_queue(self):
        """Process pending console output from the queue (main thread only)."""
        try:
            # Batch process up to 50 items per frame to avoid UI lag
            texts = []
            for _ in range(50):
                try:
                    text = self.console_queue.get_nowait()
                    texts.append(text)
                except Empty:
                    break

            if texts:
                combined = ''.join(texts)
                try:
                    current = dpg.get_value("console_output") or ""
                    # Limit console buffer to avoid memory issues with huge outputs
                    max_len = 100000
                    new_value = current + combined
                    if len(new_value) > max_len:
                        new_value = new_value[-max_len:]
                    dpg.set_value("console_output", new_value)
                except Exception:
                    pass  # UI not ready or widget doesn't exist
        except Exception:
            pass  # Ignore any queue processing errors

    def _create_input_panels(self):
        """Create input parameter panels."""
        # Design Specifications
        with dpg.collapsing_header(label=" Design Specifications", default_open=True, tag="design_header"):
            dpg.add_input_int(label="Power Rating (kVA)", tag="power_rating",
                             default_value=1000, min_value=100, max_value=100000, width=140)
            dpg.add_input_int(label="HV Voltage (V)", tag="hv_voltage",
                             default_value=20000, min_value=1000, max_value=500000, width=140)
            dpg.add_input_int(label="LV Voltage (V)", tag="lv_voltage",
                             default_value=400, min_value=100, max_value=50000, width=140)
            dpg.add_input_int(label="Frequency (Hz)", tag="frequency",
                             default_value=50, min_value=50, max_value=60, width=140)

            dpg.add_spacer(height=4)
            dpg.add_text("Winding Connections", color=(140, 145, 165))
            with dpg.group(horizontal=True):
                dpg.add_text("HV:", color=(120, 125, 145))
                dpg.add_radio_button(["Delta (D)", "Star (Y)"], tag="hv_connection",
                                    default_value="Delta (D)", horizontal=True)
            with dpg.group(horizontal=True):
                dpg.add_text("LV:", color=(120, 125, 145))
                dpg.add_radio_button(["Delta (D)", "Star (Y)"], tag="lv_connection",
                                    default_value="Star (Y)", horizontal=True)
        dpg.bind_item_theme("design_header", self.section_theme)

        # Loss Limits
        with dpg.collapsing_header(label=" Loss Limits & Constraints", default_open=True, tag="loss_header"):
            dpg.add_input_float(label="No-Load Loss (W)", tag="nll_limit",
                               default_value=900.0, min_value=0, width=140, format="%.1f")
            dpg.add_input_float(label="Load Loss (W)", tag="ll_limit",
                               default_value=7500.0, min_value=0, width=140, format="%.1f")
            dpg.add_input_float(label="Impedance (%)", tag="ucc_limit",
                               default_value=6.0, min_value=0, max_value=20, width=140, format="%.2f")
            dpg.add_input_float(label="Impedance Tolerance (%)", tag="ucc_tolerance",
                               default_value=3.0, min_value=0, max_value=50, width=140, format="%.1f")
            dpg.add_input_float(label="Loss Guarantee Factor", tag="loss_factor",
                               default_value=0.98, min_value=0.8, max_value=1.0, width=140, format="%.2f")

            # Advanced penalty options
            with dpg.tree_node(label=" Penalty Factors (Advanced)"):
                dpg.add_input_float(label="NLL Penalty", tag="penalty_nll",
                                   default_value=60.0, min_value=0, width=120, format="%.1f")
                dpg.add_input_float(label="LL Penalty", tag="penalty_ll",
                                   default_value=10.0, min_value=0, width=120, format="%.1f")
                dpg.add_input_float(label="Impedance Penalty", tag="penalty_ucc",
                                   default_value=10000.0, min_value=0, width=120, format="%.1f")
        dpg.bind_item_theme("loss_header", self.section_theme)

        # Material Selection
        with dpg.collapsing_header(label=" Material Selection", default_open=True, tag="material_header"):
            dpg.add_combo(["M5 (0.30mm)", "M4 (0.27mm)", "M3 (0.23mm)", "Hi-B (0.23mm)"],
                         label="Core Steel", tag="core_steel", default_value="M5 (0.30mm)", width=160)
            dpg.add_combo(["Aluminum", "Copper"], label="LV Conductor",
                         tag="lv_material", default_value="Copper", width=160,
                         callback=self._on_material_change)
            dpg.add_combo(["Aluminum", "Copper"], label="HV Conductor",
                         tag="hv_material", default_value="Copper", width=160,
                         callback=self._on_material_change)
            dpg.add_spacer(height=4)
            dpg.add_checkbox(label=" Circular HV Wire", tag="hv_circular",
                           default_value=False, callback=self._on_circular_wire_change)
            with dpg.tooltip(parent="hv_circular"):
                dpg.add_text("Use circular cross-section for HV wire", color=(180, 185, 200))
                dpg.add_text("(wire thickness = diameter)", color=(180, 185, 200))
        dpg.bind_item_theme("material_header", self.section_theme)

        # Parameter Constraints
        with dpg.collapsing_header(label=" Parameter Constraints", default_open=False, tag="constraints_header"):
            dpg.add_text("Lock parameters to specific values:", color=(140, 145, 165))
            dpg.add_spacer(height=4)

            # Core Diameter constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_core_dia", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_float(label="Core Diameter (mm)", tag="constraint_core_dia",
                                   default_value=200.0, min_value=80, max_value=500,
                                   width=120, format="%.1f", enabled=False)

            # Core Length constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_core_len", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_float(label="Core Length (mm)", tag="constraint_core_len",
                                   default_value=0.0, min_value=0, max_value=500,
                                   width=120, format="%.1f", enabled=False)

            # LV Turns constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_lv_turns", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_int(label="LV Turns", tag="constraint_lv_turns",
                                 default_value=30, min_value=5, max_value=100,
                                 width=120, enabled=False)

            # LV Foil Height constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_lv_height", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_float(label="LV Foil Height (mm)", tag="constraint_lv_height",
                                   default_value=400.0, min_value=200, max_value=1200,
                                   width=120, format="%.1f", enabled=False)

            # LV Foil Thickness constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_lv_thick", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_float(label="LV Foil Thickness (mm)", tag="constraint_lv_thick",
                                   default_value=1.0, min_value=0.3, max_value=4.0,
                                   width=120, format="%.2f", enabled=False)

            # HV Wire Thickness constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_hv_thick", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_float(label="HV Wire Thickness (mm)", tag="constraint_hv_thick",
                                   default_value=2.0, min_value=1.0, max_value=5.0,
                                   width=120, format="%.2f", enabled=False)

            # HV Wire Length constraint
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="", tag="lock_hv_len", default_value=False,
                               callback=self._on_constraint_toggle)
                dpg.add_input_float(label="HV Wire Length (mm)", tag="constraint_hv_len",
                                   default_value=4.0, min_value=1.0, max_value=20.0,
                                   width=120, format="%.2f", enabled=False)

            dpg.add_spacer(height=4)
            dpg.add_text("Tip: Lock parameters to optimize around specific values",
                        color=(100, 105, 125))
        dpg.bind_item_theme("constraints_header", self.section_theme)

        # Optimization Settings
        with dpg.collapsing_header(label=" Optimization Settings", default_open=True, tag="opt_header"):
            # Method selection with GPU options
            method_combo = dpg.add_combo([
                "Auto-Select (Best Available)",
                "Multi-Seed DE (CPU, Robust)",
                "Differential Evolution (CPU)",  
                "Hybrid GPU (Apple MPS)",
                "CUDA Hybrid (NVIDIA GPU)",
                "MPS GPU (Apple Silicon)",
                "MLX GPU (Apple Silicon)",
                "CUDA GPU (NVIDIA)",
                "Parallel CPU Grid Search",
                "Smart (CPU, Quick)"
            ], label="Method", tag="opt_method",
               default_value="Auto-Select (Best Available)", width=220,
               callback=self._on_method_change)

            # Tooltip for method selection - DearPyGui 2.11 compatible
            with dpg.tooltip(parent="opt_method"):
                dpg.add_text("Auto-Select: Picks best GPU", color=(180, 185, 200))
                dpg.add_text("Hybrid: 3-stage GPU + CPU", color=(180, 185, 200))
                dpg.add_text("MPS/MLX: Apple Silicon GPU", color=(180, 185, 200))
                dpg.add_text("CUDA: NVIDIA GPU", color=(180, 185, 200))
                dpg.add_text("Multi-Seed DE: 1-15 parallel seeds", color=(180, 185, 200))

            # Search depth - visible for hybrid method
            dpg.add_combo([
                "Fast (~10s)",
                "Normal (~30s)",
                "Thorough (~2-5min)",
                "Exhaustive (~10-30min)"
            ], label="Search Depth", tag="search_depth",
               default_value="Normal (~30s)", width=220, show=True)

            # Grid resolution - only for non-hybrid GPU/parallel methods
            dpg.add_combo([
                "Coarse (Fast)",
                "Medium (Balanced)",
                "Fine (Thorough)"
            ], label="Grid Resolution", tag="grid_resolution",
               default_value="Medium (Balanced)", width=220, show=False)

            # Number of seeds for Multi-Seed DE
            dpg.add_slider_int(label="DE Seeds", tag="de_seeds",
                              default_value=5, min_value=1, max_value=15, width=140, show=False)
            with dpg.tooltip(parent="de_seeds"):
                dpg.add_text("Number of parallel DE runs", color=(180, 185, 200))
                dpg.add_text("More seeds = more robust but slower", color=(180, 185, 200))

            dpg.add_input_int(label="Tolerance (%)", tag="tolerance",
                             default_value=25, min_value=0, max_value=100, width=140)
            dpg.add_checkbox(label=" Obround Core Shape", tag="obround", default_value=True)
            dpg.add_checkbox(label=" Include Cooling Ducts", tag="cooling_ducts", default_value=True)
        dpg.bind_item_theme("opt_header", self.section_theme)

    def _create_materials_panel(self):
        """Create advanced materials configuration panel."""
        with dpg.group(horizontal=True):
            # Left column - Core properties
            with dpg.child_window(width=440, height=560, border=True):
                dpg.add_text(" Core Material Properties", color=(100, 165, 255))
                dpg.add_separator()

                dpg.add_input_float(label="Core Density (g/cm3)", tag="mat_core_density",
                                   default_value=7.65, width=140, format="%.2f")
                dpg.add_input_float(label="Core Price ($/kg)", tag="mat_core_price",
                                   default_value=3.6, width=140, format="%.2f")
                dpg.add_input_float(label="Core Fill Factor (Round)", tag="mat_core_fill_round",
                                   default_value=0.84, width=140, format="%.2f")
                dpg.add_input_float(label="Core Fill Factor (Rect)", tag="mat_core_fill_rect",
                                   default_value=0.97, width=140, format="%.2f")

                dpg.add_spacer(height=8)
                dpg.add_text(" Insulation & Gaps", color=(100, 165, 255))
                dpg.add_separator()

                dpg.add_input_float(label="LV Insulation (mm)", tag="mat_lv_insulation",
                                   default_value=0.125, width=140, format="%.3f")
                dpg.add_input_float(label="HV Insulation (mm)", tag="mat_hv_insulation",
                                   default_value=0.5, width=140, format="%.2f")
                dpg.add_input_float(label="Wire Insulation (mm)", tag="mat_wire_insulation",
                                   default_value=0.12, width=140, format="%.2f")
                dpg.add_input_float(label="Main Gap (mm)", tag="mat_main_gap",
                                   default_value=12.0, width=140, format="%.1f")
                dpg.add_input_float(label="Phase Gap (mm)", tag="mat_phase_gap",
                                   default_value=12.0, width=140, format="%.1f")
                dpg.add_input_float(label="Core-LV Distance (mm)", tag="mat_core_lv_dist",
                                   default_value=2.0, width=140, format="%.1f")

                dpg.add_spacer(height=8)
                dpg.add_text(" Cooling Ducts", color=(100, 165, 255))
                dpg.add_separator()

                dpg.add_input_float(label="Duct Thickness (mm)", tag="mat_duct_thickness",
                                   default_value=4.0, width=140, format="%.1f")
                dpg.add_input_float(label="Duct Center Dist (mm)", tag="mat_duct_distance",
                                   default_value=25.0, width=140, format="%.1f")
                dpg.add_input_float(label="Duct Width (mm)", tag="mat_duct_width",
                                   default_value=7.0, width=140, format="%.1f")

            dpg.add_spacer(width=8)

            # Right column - Conductor properties
            with dpg.child_window(width=440, height=560, border=True):
                dpg.add_text(" Copper Properties", color=(255, 175, 95))
                dpg.add_separator()

                dpg.add_input_float(label="Cu Density (g/cm3)", tag="mat_cu_density",
                                   default_value=8.9, width=140, format="%.2f")
                dpg.add_input_float(label="Cu Resistivity (ohm*mm2/m)", tag="mat_cu_resistivity",
                                   default_value=0.021, width=140, format="%.4f")
                dpg.add_input_float(label="Cu Foil Price ($/kg)", tag="mat_cu_foil_price",
                                   default_value=11.55, width=140, format="%.2f")
                dpg.add_input_float(label="Cu Wire Price ($/kg)", tag="mat_cu_wire_price",
                                   default_value=11.05, width=140, format="%.2f")

                dpg.add_spacer(height=8)
                dpg.add_text(" Aluminum Properties", color=(170, 175, 210))
                dpg.add_separator()

                dpg.add_input_float(label="Al Density (g/cm3)", tag="mat_al_density",
                                   default_value=2.7, width=140, format="%.2f")
                dpg.add_input_float(label="Al Resistivity (ohm*mm2/m)", tag="mat_al_resistivity",
                                   default_value=0.0336, width=140, format="%.4f")
                dpg.add_input_float(label="Al Foil Price ($/kg)", tag="mat_al_foil_price",
                                   default_value=3.5, width=140, format="%.2f")
                dpg.add_input_float(label="Al Wire Price ($/kg)", tag="mat_al_wire_price",
                                   default_value=3.2, width=140, format="%.2f")

                dpg.add_spacer(height=8)
                dpg.add_text(" Loss Factors", color=(100, 165, 255))
                dpg.add_separator()

                dpg.add_input_float(label="Loss Factor LV", tag="mat_loss_factor_lv",
                                   default_value=1.12, width=140, format="%.2f")
                dpg.add_input_float(label="Loss Factor HV", tag="mat_loss_factor_hv",
                                   default_value=1.12, width=140, format="%.2f")

                dpg.add_spacer(height=12)
                with dpg.group(horizontal=True):
                    dpg.add_button(label=" Apply ", callback=self._apply_material_settings,
                                  width=120, height=28)
                    dpg.add_spacer(width=8)
                    dpg.add_button(label=" Reset Defaults ", callback=self._reset_material_settings,
                                  width=140, height=28)

    def _create_console_panel(self):
        """Create console output panel for viewing optimization logs."""
        with dpg.child_window(width=-1, height=560, border=True):
            with dpg.group(horizontal=True):
                dpg.add_text(" Console Output", color=(100, 165, 255))
                dpg.add_spacer(width=20)
                dpg.add_button(label=" Clear ", callback=self._clear_console, width=70, height=24)
                dpg.add_button(label=" Copy ", callback=self._copy_console, width=70, height=24)

            dpg.add_separator()

            # Console text area
            dpg.add_input_text(
                tag="console_output",
                multiline=True,
                readonly=True,
                width=-1,
                height=500,
                default_value="Console output will appear here during optimization...\n"
            )

    def _clear_console(self):
        """Clear console output."""
        dpg.set_value("console_output", "")

    def _copy_console(self):
        """Copy console output to clipboard."""
        text = dpg.get_value("console_output")
        try:
            # Try macOS pbcopy first (most reliable on macOS)
            import subprocess
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-8'))
        except Exception:
            try:
                # Fallback to pyperclip if available
                import pyperclip
                pyperclip.copy(text)
            except Exception:
                pass  # Ignore if copy fails

    def _create_results_panel(self):
        """Create the results display panel."""
        dpg.add_text(" OPTIMIZATION RESULTS", color=(100, 165, 255))
        dpg.add_separator()

        # Progress indicator section
        dpg.add_text("Progress", color=(140, 145, 165))
        dpg.add_progress_bar(tag="progress_bar", default_value=0.0, width=-1)
        dpg.bind_item_theme("progress_bar", self.progress_theme)

        with dpg.group(horizontal=True):
            dpg.add_text("Stage:", color=(100, 105, 125))
            dpg.add_text("-", tag="stage_text", color=(255, 200, 100))
            dpg.add_spacer(width=20)
            dpg.add_text("ETA:", color=(100, 105, 125))
            dpg.add_text("-", tag="eta_text", color=(120, 200, 160))

        dpg.add_text("", tag="progress_text", color=(150, 155, 175))

        dpg.add_separator()

        # Results sections
        with dpg.collapsing_header(label=" Optimal Design Parameters", tag="params_header", default_open=True):
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True, row_background=True):
                dpg.add_table_column(label="Parameter", width_fixed=True, init_width_or_weight=150)
                dpg.add_table_column(label="Value", width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(label="Unit", width_fixed=True, init_width_or_weight=50)

                params = [
                    ("Core Diameter", "core_dia", "mm"),
                    ("Core Length", "core_len", "mm"),
                    ("LV Turns", "lv_turns", "-"),
                    ("LV Foil Height", "lv_height", "mm"),
                    ("LV Foil Thickness", "lv_thick", "mm"),
                    ("HV Wire Thickness", "hv_thick", "mm"),
                ]

                for label, tag, unit in params:
                    with dpg.table_row():
                        dpg.add_text(label)
                        dpg.add_text("-", tag=f"result_{tag}")
                        dpg.add_text(unit, color=(120, 125, 145))

                # HV Wire Length row - shown only for rectangular wire
                with dpg.table_row(tag="hv_len_row"):
                    dpg.add_text("HV Wire Length", tag="hv_len_label")
                    dpg.add_text("-", tag="result_hv_len")
                    dpg.add_text("mm", color=(120, 125, 145))
        dpg.bind_item_theme("params_header", self.section_theme)

        with dpg.collapsing_header(label=" Performance Results", tag="perf_header", default_open=True):
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True, row_background=True):
                dpg.add_table_column(label="Metric", width_fixed=True, init_width_or_weight=150)
                dpg.add_table_column(label="Value", width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(label="Limit", width_fixed=True, init_width_or_weight=50)

                metrics = [
                    ("No-Load Loss", "nll", "W"),
                    ("Load Loss", "ll", "W"),
                    ("Impedance", "ucc", "%"),
                ]

                for label, tag, unit in metrics:
                    with dpg.table_row():
                        dpg.add_text(label)
                        dpg.add_text("-", tag=f"result_{tag}")
                        dpg.add_text("-", tag=f"limit_{tag}", color=(120, 125, 145))
        dpg.bind_item_theme("perf_header", self.section_theme)

        with dpg.collapsing_header(label=" Cost Breakdown", tag="cost_header", default_open=True):
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True, row_background=True):
                dpg.add_table_column(label="Component", width_fixed=True, init_width_or_weight=150)
                dpg.add_table_column(label="Weight (kg)", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="Cost ($)", width_fixed=True, init_width_or_weight=70)

                components = [
                    ("Core", "core"),
                    ("LV Winding", "lv"),
                    ("HV Winding", "hv"),
                    ("TOTAL", "total"),
                ]

                for label, tag in components:
                    with dpg.table_row():
                        if tag == "total":
                            dpg.add_text(label, color=(255, 200, 100))
                        else:
                            dpg.add_text(label)
                        dpg.add_text("-", tag=f"weight_{tag}")
                        dpg.add_text("-", tag=f"cost_{tag}")
        dpg.bind_item_theme("cost_header", self.section_theme)

        dpg.add_spacer(height=8)
        dpg.add_text("", tag="time_text", color=(120, 200, 160))

    def _on_material_change(self, sender, app_data):
        """Handle material selection change."""
        self._apply_material_settings()

    def _on_circular_wire_change(self, sender, app_data):
        """Handle circular HV wire checkbox change."""
        mainRect.HV_WIRE_CIRCULAR = app_data

    def _on_constraint_toggle(self, sender, app_data):
        """Enable/disable constraint input field when checkbox is toggled."""
        # Map checkbox tags to their corresponding input field tags
        constraint_map = {
            "lock_core_dia": "constraint_core_dia",
            "lock_core_len": "constraint_core_len",
            "lock_lv_turns": "constraint_lv_turns",
            "lock_lv_height": "constraint_lv_height",
            "lock_lv_thick": "constraint_lv_thick",
            "lock_hv_thick": "constraint_hv_thick",
            "lock_hv_len": "constraint_hv_len",
        }
        if sender in constraint_map:
            input_tag = constraint_map[sender]
            dpg.configure_item(input_tag, enabled=app_data)

    def _get_constraints(self):
        """Collect locked parameter constraints from UI."""
        constraints = {}
        constraint_params = [
            ("lock_core_dia", "constraint_core_dia", "core_diameter"),
            ("lock_core_len", "constraint_core_len", "core_length"),
            ("lock_lv_turns", "constraint_lv_turns", "lv_turns"),
            ("lock_lv_height", "constraint_lv_height", "lv_height"),
            ("lock_lv_thick", "constraint_lv_thick", "lv_thickness"),
            ("lock_hv_thick", "constraint_hv_thick", "hv_thickness"),
            ("lock_hv_len", "constraint_hv_len", "hv_length"),
        ]
        for lock_tag, value_tag, param_name in constraint_params:
            if dpg.get_value(lock_tag):
                constraints[param_name] = dpg.get_value(value_tag)
        return constraints

    def _on_method_change(self, sender, app_data):
        """Show/hide search depth or grid resolution based on method selection."""
        is_hybrid = "Hybrid" in app_data
        is_auto = "Auto-Select" in app_data
        is_gpu = "MPS" in app_data or "MLX" in app_data or "CUDA" in app_data
        is_parallel = "Parallel" in app_data
        is_de = "Differential" in app_data or "Multi-Seed" in app_data
        is_multi_de = "Multi-Seed" in app_data

        # Hybrid and Auto-Select use search_depth (Auto-Select typically uses hybrid method)
        # Other GPU/Parallel methods use grid_resolution
        # CPU methods (DE, Smart) use neither
        dpg.configure_item("search_depth", show=is_hybrid or is_auto)
        dpg.configure_item("grid_resolution", show=(is_gpu or is_parallel) and not is_hybrid and not is_auto)

        # Show DE seeds slider only for Multi-Seed DE
        dpg.configure_item("de_seeds", show=is_multi_de)

        # DE methods work better with higher tolerance (550%)
        if is_de:
            dpg.set_value("tolerance", 550)

    def _apply_material_settings(self):
        """Apply material settings to mainRect module."""
        # Core properties
        mainRect.CoreDensity = dpg.get_value("mat_core_density")
        mainRect.CorePricePerKg = dpg.get_value("mat_core_price")
        mainRect.CoreFillingFactorRound = dpg.get_value("mat_core_fill_round")
        mainRect.CoreFillingFactorRectangular = dpg.get_value("mat_core_fill_rect")

        # Insulation
        mainRect.LVInsulationThickness = dpg.get_value("mat_lv_insulation")
        mainRect.HVInsulationThickness = dpg.get_value("mat_hv_insulation")
        mainRect.INSULATION_THICKNESS_WIRE = dpg.get_value("mat_wire_insulation")
        mainRect.MainGap = dpg.get_value("mat_main_gap")
        mainRect.PhaseGap = dpg.get_value("mat_phase_gap")
        mainRect.DistanceCoreLV = dpg.get_value("mat_core_lv_dist")

        # Cooling ducts
        mainRect.COOLING_DUCT_THICKNESS = dpg.get_value("mat_duct_thickness")
        mainRect.COOLING_DUCT_CENTER_TO_CENTER_DISTANCE = dpg.get_value("mat_duct_distance")
        mainRect.COOLING_DUCT_WIDTH = dpg.get_value("mat_duct_width")

        # Conductor densities
        mainRect.CuDensity = dpg.get_value("mat_cu_density")
        mainRect.AlDensity = dpg.get_value("mat_al_density")

        # Loss factors
        mainRect.AdditionalLossFactorLV = dpg.get_value("mat_loss_factor_lv")
        mainRect.AdditionalLossFactorHV = dpg.get_value("mat_loss_factor_hv")

        # Set conductor properties based on selection
        lv_mat = dpg.get_value("lv_material")
        hv_mat = dpg.get_value("hv_material")

        if lv_mat == "Copper":
            mainRect.materialToBeUsedFoil_Density = dpg.get_value("mat_cu_density")
            mainRect.materialToBeUsedFoil_Price = dpg.get_value("mat_cu_foil_price")
            mainRect.materialToBeUsedFoil_Resistivity = dpg.get_value("mat_cu_resistivity")
        else:
            mainRect.materialToBeUsedFoil_Density = dpg.get_value("mat_al_density")
            mainRect.materialToBeUsedFoil_Price = dpg.get_value("mat_al_foil_price")
            mainRect.materialToBeUsedFoil_Resistivity = dpg.get_value("mat_al_resistivity")

        if hv_mat == "Copper":
            mainRect.materialToBeUsedWire_Density = dpg.get_value("mat_cu_density")
            mainRect.materialToBeUsedWire_Price = dpg.get_value("mat_cu_wire_price")
            mainRect.materialToBeUsedWire_Resistivity = dpg.get_value("mat_cu_resistivity")
        else:
            mainRect.materialToBeUsedWire_Density = dpg.get_value("mat_al_density")
            mainRect.materialToBeUsedWire_Price = dpg.get_value("mat_al_wire_price")
            mainRect.materialToBeUsedWire_Resistivity = dpg.get_value("mat_al_resistivity")

    def _reset_material_settings(self):
        """Reset material settings to defaults."""
        # Core
        dpg.set_value("mat_core_density", 7.65)
        dpg.set_value("mat_core_price", 3.6)
        dpg.set_value("mat_core_fill_round", 0.84)
        dpg.set_value("mat_core_fill_rect", 0.97)

        # Insulation
        dpg.set_value("mat_lv_insulation", 0.125)
        dpg.set_value("mat_hv_insulation", 0.5)
        dpg.set_value("mat_wire_insulation", 0.12)
        dpg.set_value("mat_main_gap", 12.0)
        dpg.set_value("mat_phase_gap", 12.0)
        dpg.set_value("mat_core_lv_dist", 2.0)

        # Cooling ducts
        dpg.set_value("mat_duct_thickness", 4.0)
        dpg.set_value("mat_duct_distance", 25.0)
        dpg.set_value("mat_duct_width", 7.0)

        # Copper
        dpg.set_value("mat_cu_density", 8.9)
        dpg.set_value("mat_cu_resistivity", 0.021)
        dpg.set_value("mat_cu_foil_price", 11.55)
        dpg.set_value("mat_cu_wire_price", 11.05)

        # Aluminum
        dpg.set_value("mat_al_density", 2.7)
        dpg.set_value("mat_al_resistivity", 0.0336)
        dpg.set_value("mat_al_foil_price", 3.5)
        dpg.set_value("mat_al_wire_price", 3.2)

        # Loss factors
        dpg.set_value("mat_loss_factor_lv", 1.12)
        dpg.set_value("mat_loss_factor_hv", 1.12)

        self._apply_material_settings()

    def _update_mainrect_params(self):
        """Update mainRect module parameters from UI inputs."""
        # Apply material settings first
        self._apply_material_settings()

        # HV wire shape (circular vs rectangular)
        mainRect.HV_WIRE_CIRCULAR = dpg.get_value("hv_circular")

        # Connection types (Delta=1.0, Star=1/sqrt(3) for phase voltage)
        hv_conn = dpg.get_value("hv_connection")
        lv_conn = dpg.get_value("lv_connection")
        mainRect.HVCONNECTION = 1.0 if "Delta" in hv_conn else (1.0 / 1.732)
        mainRect.LVCONNECTION = 1.0 if "Delta" in lv_conn else (1.0 / 1.732)

        # Power and voltage
        mainRect.POWERRATING = dpg.get_value("power_rating")
        mainRect.HVRATE = dpg.get_value("hv_voltage") * mainRect.HVCONNECTION
        mainRect.LVRATE = dpg.get_value("lv_voltage") * mainRect.LVCONNECTION
        mainRect.FREQUENCY = dpg.get_value("frequency")

        # Loss limits
        loss_factor = dpg.get_value("loss_factor")
        mainRect.GUARANTEED_NO_LOAD_LOSS = dpg.get_value("nll_limit") * loss_factor
        mainRect.GUARANTEED_LOAD_LOSS = dpg.get_value("ll_limit") * loss_factor
        mainRect.GUARANTEED_UCC = dpg.get_value("ucc_limit")
        mainRect.UCC_TOLERANCE_PERCENT = dpg.get_value("ucc_tolerance")
        mainRect.UCC_TOLERANCE = mainRect.GUARANTEED_UCC * (mainRect.UCC_TOLERANCE_PERCENT / 100)

        # Penalties
        mainRect.PENALTY_NLL_FACTOR = dpg.get_value("penalty_nll")
        mainRect.PENALTY_LL_FACTOR = dpg.get_value("penalty_ll")
        mainRect.PENALTY_UCC_FACTOR = dpg.get_value("penalty_ucc")

    def _format_eta(self, seconds):
        """Format ETA seconds into human readable string."""
        if seconds is None or seconds < 0:
            return "-"
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _run_optimization(self):
        """Start the optimization in a background thread."""
        if self.is_optimizing:
            return

        self.is_optimizing = True
        self.duck_angle = 0  # Reset duck animation
        self._last_duck_update = time.time()
        dpg.configure_item("run_btn", enabled=False)
        dpg.configure_item("stop_btn", enabled=True)
        dpg.set_value("status_text", "Optimizing...")
        dpg.configure_item("status_text", color=(255, 200, 100))
        dpg.set_value("progress_bar", 0.0)
        dpg.set_value("progress_text", "Starting optimization...")
        dpg.set_value("stage_text", "-")
        dpg.set_value("eta_text", "-")

        # Clear and setup console capture with thread-safe queue
        self._clear_console()
        # Clear any stale items from queue
        while not self.console_queue.empty():
            try:
                self.console_queue.get_nowait()
            except Empty:
                break
        self.console_capture = ConsoleCapture(output_queue=self.console_queue)
        # Start capture EARLY so all prints are captured (including auto-select messages)
        self.console_capture.start_capture()

        # Update parameters
        self._update_mainrect_params()

        # Get optimization settings
        method_str = dpg.get_value("opt_method")
        if "Auto-Select" in method_str:
            # Auto-detect best available method
            if mainRect.CUDA_AVAILABLE:
                method = 'cuda_hybrid'
                print("Auto-selected: CUDA Hybrid (NVIDIA GPU detected)")
            elif mainRect.MPS_AVAILABLE:
                method = 'hybrid'
                print("Auto-selected: MPS Hybrid (Apple Silicon detected)")
            elif mainRect.MLX_AVAILABLE:
                method = 'mlx'
                print("Auto-selected: MLX GPU (Apple Silicon detected)")
            else:
                method = 'de'
                print("Auto-selected: Differential Evolution (No GPU detected)")
        elif "CUDA Hybrid" in method_str:
            method = 'cuda_hybrid'
        elif "Hybrid" in method_str:
            method = 'hybrid'
        elif "MPS" in method_str:
            method = 'mps'
        elif "MLX" in method_str:
            method = 'mlx'
        elif "CUDA" in method_str:
            method = 'gpu'
        elif "Multi-Seed" in method_str:
            method = 'multi_de'
        elif "Differential" in method_str:
            method = 'de'
        elif "Parallel" in method_str:
            method = 'parallel'
        else:
            method = 'smart'

        # Get search depth for hybrid method
        depth_str = dpg.get_value("search_depth")
        if "Fast" in depth_str:
            search_depth = 'fast'
        elif "Thorough" in depth_str:
            search_depth = 'thorough'
        elif "Exhaustive" in depth_str:
            search_depth = 'exhaustive'
        else:
            search_depth = 'normal'

        # Get grid resolution for GPU/parallel methods
        grid_res_str = dpg.get_value("grid_resolution")
        if "Coarse" in grid_res_str:
            grid_resolution = 'coarse'
        elif "Fine" in grid_res_str:
            grid_resolution = 'fine'
        else:
            grid_resolution = 'medium'

        tolerance = dpg.get_value("tolerance")
        obround = dpg.get_value("obround")
        cooling_ducts = dpg.get_value("cooling_ducts")
        de_seeds = dpg.get_value("de_seeds")
        constraints = self._get_constraints()

        def progress_callback(stage, progress, message, eta=None):
            """Handle progress updates from optimization."""
            # Check if stop was requested - raise exception to abort
            if not self.is_optimizing:
                raise InterruptedError("Optimization stopped by user")

            try:
                # Update progress bar (0-1 scale)
                dpg.set_value("progress_bar", min(max(progress, 0.0), 1.0))

                # Update stage text
                if isinstance(stage, int):
                    stage_names = {1: "Stage 1/3: Coarse GPU", 2: "Stage 2/3: Fine GPU", 3: "Stage 3/3: Local CPU"}
                    stage_text = stage_names.get(stage, f"Stage {stage}")
                else:
                    stage_text = str(stage)
                dpg.set_value("stage_text", stage_text)

                # Update ETA
                eta_text = self._format_eta(eta)
                dpg.set_value("eta_text", eta_text)

                # Update progress message
                dpg.set_value("progress_text", message)
            except InterruptedError:
                raise  # Re-raise stop signal
            except Exception:
                pass  # Ignore other UI update errors

        def run_opt():
            try:
                method_name = method.upper()
                if method == 'hybrid':
                    method_name = f"MPS HYBRID ({search_depth.upper()})"
                elif method == 'cuda_hybrid':
                    method_name = f"CUDA HYBRID ({search_depth.upper()})"
                dpg.set_value("progress_text", f"Running {method_name} optimization...")

                # Print constraint info
                if constraints:
                    print(f"\nParameter Constraints Active:")
                    for param, value in constraints.items():
                        print(f"  - {param}: {value}")
                    print()
                dpg.set_value("progress_bar", 0.05)

                # Run the optimization with progress callback
                result = mainRect.StartFast(
                    tolerance=tolerance,
                    obround=obround,
                    put_cooling_ducts=cooling_ducts,
                    method=method,
                    print_result=True,  # Enable prints for console output
                    grid_resolution=grid_resolution,
                    search_depth=search_depth,
                    progress_callback=progress_callback,
                    n_seeds=de_seeds,
                    constraints=constraints
                )

                dpg.set_value("progress_bar", 1.0)
                dpg.set_value("stage_text", "Complete")
                dpg.set_value("eta_text", "-")

                if result:
                    self.result = result
                    self._display_results(result)
                    dpg.set_value("status_text", "Complete")
                    dpg.configure_item("status_text", color=(72, 199, 142))
                    dpg.set_value("progress_text", "Optimization complete!")
                    try:
                        play_sound(resource_path("quack.wav"))
                    except:
                        pass
                else:
                    dpg.set_value("status_text", "No valid design found")
                    dpg.configure_item("status_text", color=(255, 99, 99))
                    dpg.set_value("progress_text", "No valid design found within constraints.")

            except InterruptedError:
                # User requested stop - handle gracefully
                dpg.set_value("status_text", "Stopped")
                dpg.configure_item("status_text", color=(255, 150, 100))
                dpg.set_value("progress_text", "Optimization stopped by user")
                dpg.set_value("progress_bar", 0.0)
                dpg.set_value("stage_text", "Stopped")
                dpg.set_value("eta_text", "-")
                self.console_queue.put_nowait("\n--- Optimization stopped by user ---\n")

            except Exception as e:
                dpg.set_value("status_text", "Error")
                dpg.configure_item("status_text", color=(255, 99, 99))
                dpg.set_value("progress_text", f"Error: {str(e)}")
                import traceback
                try:
                    self.console_queue.put_nowait(f"\nError: {str(e)}\n{traceback.format_exc()}")
                except:
                    pass
            finally:
                # Stop console capture
                if self.console_capture:
                    self.console_capture.stop_capture()
                self.is_optimizing = False
                self.duck_angle = 0  # Frame callback will reset duck in main thread
                dpg.configure_item("run_btn", enabled=True)
                dpg.configure_item("stop_btn", enabled=False)

        self.optimization_thread = threading.Thread(target=run_opt, daemon=True)
        self.optimization_thread.start()

    def _stop_optimization(self):
        """Stop the optimization (note: can't truly stop mid-calculation)."""
        self.is_optimizing = False
        # Clean up console capture
        if self.console_capture:
            self.console_capture.stop_capture()
        dpg.set_value("status_text", "Stopped")
        dpg.configure_item("status_text", color=(255, 150, 100))
        dpg.set_value("stage_text", "Stopped")
        dpg.set_value("eta_text", "-")
        dpg.configure_item("run_btn", enabled=True)
        dpg.configure_item("stop_btn", enabled=False)

    def _display_results(self, result):
        """Display optimization results in the UI."""
        # Design parameters
        dpg.set_value("result_core_dia", f"{result['core_diameter']:.1f}")
        dpg.set_value("result_core_len", f"{result['core_length']:.1f}")
        dpg.set_value("result_lv_turns", f"{result['lv_turns']:.0f}")
        dpg.set_value("result_lv_height", f"{result['lv_height']:.1f}")
        dpg.set_value("result_lv_thick", f"{result['lv_thickness']:.2f}")
        dpg.set_value("result_hv_thick", f"{result['hv_thickness']:.2f}")

        # For circular wire, hide HV Wire Length row (only diameter matters)
        is_circular = mainRect.HV_WIRE_CIRCULAR
        if is_circular:
            dpg.hide_item("hv_len_row")
        else:
            dpg.show_item("hv_len_row")
            dpg.set_value("result_hv_len", f"{result['hv_length']:.1f}")

        # Performance metrics
        dpg.set_value("result_nll", f"{result['no_load_loss']:.1f}")
        dpg.set_value("result_ll", f"{result['load_loss']:.1f}")
        dpg.set_value("result_ucc", f"{result['impedance']:.2f}")

        # Limits
        dpg.set_value("limit_nll", f"{mainRect.GUARANTEED_NO_LOAD_LOSS:.0f}")
        dpg.set_value("limit_ll", f"{mainRect.GUARANTEED_LOAD_LOSS:.0f}")
        dpg.set_value("limit_ucc", f"{mainRect.GUARANTEED_UCC:.1f}")

        # Weights and costs
        dpg.set_value("weight_core", f"{result['core_weight']:.1f}")
        dpg.set_value("weight_lv", f"{result['lv_weight']:.1f}")
        dpg.set_value("weight_hv", f"{result['hv_weight']:.1f}")
        total_weight = result['core_weight'] + result['lv_weight'] + result['hv_weight']
        dpg.set_value("weight_total", f"{total_weight:.1f}")

        dpg.set_value("cost_core", f"${result['core_price']:.2f}")
        dpg.set_value("cost_lv", f"${result['lv_price']:.2f}")
        dpg.set_value("cost_hv", f"${result['hv_price']:.2f}")
        # Show actual material cost (sum of components), not penalized optimization price
        total_cost = result['core_price'] + result['lv_price'] + result['hv_price']
        dpg.set_value("cost_total", f"${total_cost:.2f}")

        # Time
        dpg.set_value("time_text", f"Optimization completed in {result['time']:.2f} seconds")

    def run(self):
        """Run the application main loop."""
        while dpg.is_dearpygui_running():
            # Process console queue directly in main loop for reliable updates
            self._process_console_queue()

            # Animate duck when optimizing
            if self.is_optimizing:
                current_time = time.time()
                if current_time - self._last_duck_update > 0.05:
                    self.duck_angle += 0.15
                    self._draw_duck(30, 22, self.duck_angle)
                    self._last_duck_update = current_time
            elif self.duck_angle != 0:
                self.duck_angle = 0
                self._draw_duck(30, 22, 0)

            dpg.render_dearpygui_frame()

        dpg.destroy_context()


def main():
    app = TransformerOptimizerApp()
    app.run()


if __name__ == "__main__":
    main()
