import tkinter as tk
from tkinter import ttk, messagebox
import colorsys
import json
import os

# --- DARK THEME PALETTE (IDE Style) ---
BG_MAIN = "#1e1e1e"
BG_PANEL = "#252526"
BG_INPUT = "#3c3c3c"
COLOR_PRIMARY = "#0e639c"
COLOR_PRIMARY_HOVER = "#1177bb"
COLOR_DANGER = "#c94a4d"
COLOR_WALL = "#666666"
COLOR_ERASER = "#333333"
TEXT_LIGHT = "#cccccc"
TEXT_WHITE = "#ffffff"
CANVAS_BG = "#121212"
GRID_COLOR = "#2a2a2a"
AXIS_COLOR = "#555555"

FONT_MAIN = ("Helvetica", 10)
FONT_TITLE = ("Helvetica", 11, "bold")

class HSVPicker(tk.Toplevel):
    def __init__(self, parent, initial_color_hex, callback):
        super().__init__(parent)
        self.title("HSV Color Picker")
        self.configure(bg=BG_PANEL)
        self.callback = callback
        self.width = 360
        self.height = 100

        # Convert initial HEX to HSV for state management
        r = int(initial_color_hex[1:3], 16) / 255.0
        g = int(initial_color_hex[3:5], 16) / 255.0
        b = int(initial_color_hex[5:7], 16) / 255.0
        self.h, self.s, self.v = colorsys.rgb_to_hsv(r, g, b)

        self.setup_ui()
        self.draw_hs_gradient()
        self.update_preview()

    def setup_ui(self):
        tk.Label(self, text="Hue (X) and Saturation (Y)", font=FONT_TITLE, bg=BG_PANEL, fg=TEXT_WHITE).pack(pady=(15, 5))

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, cursor="crosshair", 
                                bg=BG_MAIN, highlightthickness=1, highlightbackground=BG_INPUT)
        self.canvas.pack(padx=20, pady=5)

        self.hs_img = tk.PhotoImage(width=self.width, height=self.height)
        self.canvas.create_image(0, 0, image=self.hs_img, anchor="nw")
        self.cursor_id = self.canvas.create_oval(0, 0, 0, 0, outline="black", width=2)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_click)

        self.preview = tk.Label(self, text="Preview", width=20, height=2, font=FONT_TITLE)

        v_frame = tk.Frame(self, bg=BG_PANEL)
        v_frame.pack(fill="x", padx=20, pady=15)
        tk.Label(v_frame, text="Value (V): ", font=FONT_MAIN, bg=BG_PANEL, fg=TEXT_LIGHT).pack(side="left")
        self.v_scale = ttk.Scale(v_frame, from_=0.0, to=1.0, orient="horizontal", command=self.on_v_change)
        self.v_scale.set(self.v) 
        self.v_scale.pack(side="left", fill="x", expand=True, padx=10)

        self.preview.pack(pady=10)

        btn_confirm = tk.Button(self, text="Confirm Color", bg=COLOR_PRIMARY, fg="white", font=FONT_TITLE,
                                relief="flat", bd=0, cursor="hand2", padx=15, pady=8, command=self.confirm)
        btn_confirm.pack(pady=(5, 20))

    def draw_hs_gradient(self):
        # Generate color data for PhotoImage
        rows = []
        for y in range(self.height):
            s = 1.0 - (y / self.height)
            row = []
            for x in range(self.width):
                h = x / self.width
                r, g, b = colorsys.hsv_to_rgb(h, s, 1.0) 
                row.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
            rows.append("{" + " ".join(row) + "}")
        self.hs_img.put(" ".join(rows))

        x = int(self.h * self.width)
        y = int((1.0 - self.s) * self.height)
        self.canvas.coords(self.cursor_id, x-5, y-5, x+5, y+5)

    def on_canvas_click(self, event):
        x = max(0, min(self.width, event.x))
        y = max(0, min(self.height, event.y))
        self.canvas.coords(self.cursor_id, x-5, y-5, x+5, y+5)
        self.h = x / self.width
        self.s = 1.0 - (y / self.height)
        self.update_preview()

    def on_v_change(self, val):
        self.v = float(val)
        self.update_preview()

    def update_preview(self):
        r, g, b = colorsys.hsv_to_rgb(self.h, self.s, self.v)
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        # Calculate optimal text contrast
        luminance = (0.299*r + 0.587*g + 0.114*b)
        text_color = "black" if luminance > 0.5 else "white"
        self.preview.config(bg=hex_color, fg=text_color)
        self.current_hex = hex_color

    def confirm(self):
        self.callback(self.current_hex)
        self.destroy()


class WorldBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic World Builder Pro")
        self.root.configure(bg=BG_MAIN)

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_PANEL, foreground=TEXT_WHITE, bordercolor=BG_PANEL, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", BG_INPUT)])
        style.configure("TSpinbox", fieldbackground=BG_INPUT, background=BG_PANEL, foreground=TEXT_WHITE, bordercolor=BG_PANEL, padding=5)

        self.ui_cell_size = 30 
        self.cols = 30
        self.rows = 30
        self.current_tool = "wall"
        self.world_data = {}  # Stores grid coordinate state: {(row, col): {data}}
        self.selected_color_hex = COLOR_PRIMARY
        self.selected_shape = tk.StringVar(value="cube")

        self.setup_ui()
        self.draw_grid()

    def create_flat_button(self, parent, text, bg, fg, command, bold=False):
        font = FONT_TITLE if bold else FONT_MAIN
        btn = tk.Button(parent, text=text, bg=bg, fg=fg, font=font, 
                        relief="flat", activebackground=bg, activeforeground=fg, bd=0, padx=10, pady=5, cursor="hand2", command=command)
        return btn

    def setup_ui(self):
        toolbar = tk.Frame(self.root, bg=BG_PANEL, padx=15, pady=15, highlightthickness=1, highlightbackground=BG_MAIN)
        toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)

        # --- MAP SECTION ---
        tk.Label(toolbar, text="Map Settings", font=FONT_TITLE, bg=BG_PANEL, fg=TEXT_WHITE).pack(pady=(0, 10))
        map_frame = tk.Frame(toolbar, bg=BG_PANEL)
        map_frame.pack(fill="x", pady=5)

        tk.Label(map_frame, text="Width (X):", bg=BG_PANEL, fg=TEXT_LIGHT, font=FONT_MAIN).grid(row=0, column=0, sticky="w", pady=2)
        self.width_spin = ttk.Spinbox(map_frame, from_=10, to=100, width=6)
        self.width_spin.insert(0, str(self.cols)); self.width_spin.delete(1, 'end')
        self.width_spin.grid(row=0, column=1, pady=2, padx=5)

        tk.Label(map_frame, text="Height (Z):", bg=BG_PANEL, fg=TEXT_LIGHT, font=FONT_MAIN).grid(row=1, column=0, sticky="w", pady=2)
        self.height_spin = ttk.Spinbox(map_frame, from_=10, to=100, width=6)
        self.height_spin.insert(0, str(self.rows)); self.height_spin.delete(1, 'end')
        self.height_spin.grid(row=1, column=1, pady=2, padx=5)

        tk.Label(map_frame, text="Scale (m/cell):", bg=BG_PANEL, fg=TEXT_LIGHT, font=FONT_MAIN).grid(row=2, column=0, sticky="w", pady=2)
        self.scale_spin = ttk.Spinbox(map_frame, from_=0.1, to=10.0, increment=0.1, format="%.1f", width=6)
        self.scale_spin.insert(0, "1.0"); self.scale_spin.delete(1, 'end')
        self.scale_spin.grid(row=2, column=1, pady=2, padx=5)

        self.create_flat_button(toolbar, "Apply Dimensions", BG_INPUT, TEXT_WHITE, self.resize_map).pack(fill="x", pady=5)
        tk.Frame(toolbar, height=1, bg=BG_INPUT).pack(fill="x", pady=15)

        # --- TOOLS SECTION ---
        tk.Label(toolbar, text="Basic Tools", font=FONT_TITLE, bg=BG_PANEL, fg=TEXT_WHITE).pack(pady=(0, 10))
        self.btn_wall = self.create_flat_button(toolbar, "Draw Wall", COLOR_WALL, "white", lambda: self.select_tool("wall"))
        self.btn_wall.pack(fill="x", pady=2)

        self.btn_eraser = tk.Button(toolbar, text="Eraser", bg=BG_MAIN, fg=TEXT_LIGHT, font=FONT_MAIN, 
                                    relief="solid", bd=1, padx=10, pady=5, cursor="hand2", command=lambda: self.select_tool("eraser"))
        self.btn_eraser.pack(fill="x", pady=5)

        tk.Frame(toolbar, height=1, bg=BG_INPUT).pack(fill="x", pady=15)

        # --- OBJECTS SECTION ---
        tk.Label(toolbar, text="Build Object", font=FONT_TITLE, bg=BG_PANEL, fg=TEXT_WHITE).pack(pady=(0, 10))

        tk.Label(toolbar, text="Shape:", bg=BG_PANEL, fg=TEXT_LIGHT, font=FONT_MAIN).pack(anchor="w")
        shapes = ["cube", "sphere", "cylinder", "pyramid", "parallelepiped"]
        self.shape_combo = ttk.Combobox(toolbar, textvariable=self.selected_shape, values=shapes, state="readonly")
        self.shape_combo.pack(fill="x", pady=(0, 10))

        tk.Label(toolbar, text="Color:", bg=BG_PANEL, fg=TEXT_LIGHT, font=FONT_MAIN).pack(anchor="w")
        self.color_btn = tk.Button(toolbar, text="Open Color Picker", bg=self.selected_color_hex, fg="white", font=FONT_MAIN,
                                   relief="flat", bd=0, cursor="hand2", pady=5, command=self.open_hsv_picker)
        self.color_btn.pack(fill="x", pady=5)

        self.btn_object = self.create_flat_button(toolbar, "Use this Object", BG_INPUT, TEXT_WHITE, lambda: self.select_tool("object"), bold=True)
        self.btn_object.pack(fill="x", pady=10)

        tk.Frame(toolbar, height=1, bg=BG_INPUT).pack(fill="x", pady=15)

        # --- ACTIONS SECTION ---
        self.create_flat_button(toolbar, "Export JSON", COLOR_PRIMARY, "white", self.export_json, bold=True).pack(fill="x", pady=5)
        self.create_flat_button(toolbar, "Clear Map", COLOR_DANGER, "white", self.clear_grid).pack(fill="x", pady=5)

        self.select_tool("wall")

        # --- CANVAS ---
        self.canvas_frame = tk.Frame(self.root, bg=BG_MAIN)
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.create_canvas()

    def create_canvas(self):
        if hasattr(self, 'canvas'):
            self.canvas.destroy()

        canvas_width = self.cols * self.ui_cell_size
        canvas_height = self.rows * self.ui_cell_size

        self.canvas = tk.Canvas(self.canvas_frame, width=canvas_width, height=canvas_height, bg=CANVAS_BG, 
                                highlightthickness=1, highlightbackground=BG_INPUT)
        self.canvas.pack(anchor="center")

        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.draw_grid()

    def resize_map(self):
        try:
            self.cols = int(self.width_spin.get())
            self.rows = int(self.height_spin.get())
            self.create_canvas()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")

    def open_hsv_picker(self):
        HSVPicker(self.root, self.selected_color_hex, self.on_color_selected)

    def on_color_selected(self, hex_color):
        self.selected_color_hex = hex_color
        # Recalculate button text color for readability
        r, g, b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
        text_color = "black" if (0.299*r + 0.587*g + 0.114*b) > 128 else "white"
        self.color_btn.config(bg=self.selected_color_hex, fg=text_color)
        self.select_tool("object")

    def select_tool(self, tool_name):
        self.current_tool = tool_name

        self.btn_wall.config(bd=0, bg=COLOR_WALL, fg="white")
        self.btn_eraser.config(bd=1, bg=BG_MAIN, fg=TEXT_LIGHT)
        self.btn_object.config(bd=0, bg=BG_INPUT, fg=TEXT_WHITE)

        if tool_name == "wall": 
            self.btn_wall.config(bd=2, bg="#888888")
        elif tool_name == "eraser": 
            self.btn_eraser.config(bd=2, bg="#555555", fg="white")
        elif tool_name == "object": 
            self.btn_object.config(bd=2, bg=COLOR_PRIMARY, fg="white")

    def paint(self, event):
        col = event.x // self.ui_cell_size
        row = event.y // self.ui_cell_size

        if 0 <= col < self.cols and 0 <= row < self.rows:
            x1 = col * self.ui_cell_size
            y1 = row * self.ui_cell_size
            x2 = x1 + self.ui_cell_size
            y2 = y1 + self.ui_cell_size

            # Clear existing cell
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=CANVAS_BG, outline=GRID_COLOR)
            if (row, col) in self.world_data:
                del self.world_data[(row, col)]

            self.draw_grid()

            # Apply tools
            if self.current_tool == "wall":
                self.world_data[(row, col)] = {"type": "wall", "color": COLOR_WALL}
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=COLOR_WALL, outline=BG_MAIN)

            elif self.current_tool == "object":
                shape = self.selected_shape.get()
                self.world_data[(row, col)] = {"type": "object", "shape": shape, "color": self.selected_color_hex}

                if shape == "sphere":
                    self.canvas.create_oval(x1+2, y1+2, x2-2, y2-2, fill=self.selected_color_hex, outline="white", width=1)
                else:
                    self.canvas.create_rectangle(x1+2, y1+2, x2-2, y2-2, fill=self.selected_color_hex, outline="white", width=1)

                r, g, b = int(self.selected_color_hex[1:3],16), int(self.selected_color_hex[3:5],16), int(self.selected_color_hex[5:7],16)
                text_color = "black" if (0.299*r + 0.587*g + 0.114*b) > 128 else "white"
                label = shape[:2].capitalize()
                self.canvas.create_text(
                    x1 + self.ui_cell_size//2, y1 + self.ui_cell_size//2, 
                    text=label, fill=text_color, font=("Helvetica", max(8, self.ui_cell_size//3), "bold")
                )

    def draw_grid(self):
        for i in range(self.cols + 1):
            x = i * self.ui_cell_size
            self.canvas.create_line(x, 0, x, self.rows * self.ui_cell_size, fill=GRID_COLOR)
        for i in range(self.rows + 1):
            y = i * self.ui_cell_size
            self.canvas.create_line(0, y, self.cols * self.ui_cell_size, y, fill=GRID_COLOR)

        # Draw coordinate axes (center)
        center_x = (self.cols // 2) * self.ui_cell_size
        center_y = (self.rows // 2) * self.ui_cell_size
        self.canvas.create_line(0, center_y, self.cols * self.ui_cell_size, center_y, fill="red", width=3)
        self.canvas.create_line(center_x, 0, center_x, self.rows * self.ui_cell_size, fill="green", width=3)

    def clear_grid(self):
        self.world_data.clear()
        self.canvas.delete("all")
        self.draw_grid()

    def export_json(self):
        output = {"walls": [], "objects": []}
        offset_x = self.cols // 2
        offset_y = self.rows // 2
        wall_counter = 0
        obj_counter = 0

        try: cell_scale = float(self.scale_spin.get())
        except ValueError: cell_scale = 1.0 

        for (row, col), data in self.world_data.items():
            # Coordinate conversion: map index to world space relative to center
            sim_x = float(col - offset_x) * cell_scale
            sim_y = -float(row - offset_y) * cell_scale

            hex_color = data["color"].lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = [rgb[0], rgb[1], rgb[2], 255]

            if data["type"] == "wall":
                output["walls"].append({
                    "name": f"wall_{wall_counter}",
                    "x": sim_x,
                    "y": sim_y,
                    "z": 1.0,
                    "width": cell_scale,
                    "height": 2.0,
                    "depth": cell_scale,
                    "color": rgba_color
                })
                wall_counter += 1
            else:
                output["objects"].append({
                    "name": f"{data['shape']}_{obj_counter}",
                    "shape": data["shape"],
                    "x": sim_x,
                    "y": sim_y,
                    "z": cell_scale * 0.5,
                    "size": cell_scale,
                    "color": rgba_color
                })
                obj_counter += 1

        output_path = "/workspace/worlds/world_config.json"
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=4)
            print(f"World exported to {output_path} (Scale: {cell_scale} m/cell)!")
        except Exception as e:
            print(f"Error during save: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = WorldBuilder(root)
    root.mainloop()
