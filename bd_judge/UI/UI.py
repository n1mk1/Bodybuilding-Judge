import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from xframe import analyze_xframe
from conditioning import (
    AnalysisConfig, 
    PoseAnalyzer, 
    RegionExtractor, 
    ConditioningAnalyzer,
    ResultVisualizer
)
from poseClassifier import ImageOrganizer, AthleteLoader


class BodybuildingCompetitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bodybuilding Competition Analysis")
        
        window_width = 1600
        window_height = 900
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        self.bg_color = "#202020"
        self.panel_bg = "#2d2d2d"
        self.accent_color = "#7f6df2"
        self.button_color = "#7f6df2"
        self.root.configure(bg=self.bg_color)
        
        self.working_directory = os.getcwd()
        self.athletes = {}
        self.current_athletes = [None, None]
        self.current_pose = "front_double_bicep"
        self.analysis_window = None
        
        self.pose_map = {
            "Front Double Bicep": "front_double_bicep",
            "Back Double Bicep": "back_double_bicep"
        }
        
        self.setup_ui()

    def setup_ui(self):
        main = tk.Frame(self.root, bg=self.bg_color)
        main.pack(fill=tk.BOTH, expand=True)
        main.grid_columnconfigure(0, weight=4)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)
        
        left = tk.Frame(main, bg=self.bg_color)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        self._build_log_panel(main)
        self._build_header(left)
        self._build_comparison_area(left)
        self._build_analysis_controls(left)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=self.panel_bg, relief=tk.GROOVE, bd=2)
        header.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            header, 
            text="Competition Analysis Tool",
            font=("Cascadia Code", 16, "bold"),
            bg=self.panel_bg,
            fg=self.accent_color
        ).pack(pady=10)
        
        controls = tk.Frame(header, bg=self.panel_bg)
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        dir_section = tk.Frame(controls, bg=self.panel_bg)
        dir_section.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(dir_section, text="Directory:", font=("Cascadia Code", 10), bg=self.panel_bg, fg="#dcddde").pack(side=tk.LEFT, padx=(0, 5))
        
        self.dir_label = tk.Label(
            dir_section, 
            text=self._truncate(self.working_directory, 60),
            font=("Cascadia Code", 9),
            bg=self.panel_bg,
            fg="#dcddde",
            anchor="w"
        )
        self.dir_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Button(
            dir_section,
            text="Browse...",
            command=self.browse_directory,
            font=("Cascadia Code", 9),
            bg="#7f6df2",
            fg="white",
            relief=tk.RAISED,
            padx=12,
            pady=4
        ).pack(side=tk.LEFT)
        
        buttons = tk.Frame(controls, bg=self.panel_bg)
        buttons.pack(side=tk.RIGHT, padx=10)
        
        tk.Button(
            buttons,
            text="Organize Images",
            command=self.organize_poses,
            font=("Cascadia Code", 10),
            bg=self.button_color,
            fg="white",
            relief=tk.RAISED,
            padx=12,
            pady=5
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Button(
            buttons,
            text="Load Comparison",
            command=self.load_battle,
            font=("Cascadia Code", 10),
            bg=self.button_color,
            fg="white",
            relief=tk.RAISED,
            padx=12,
            pady=5
        ).pack(side=tk.LEFT, padx=3)

    def _build_comparison_area(self, parent):
        comp = tk.Frame(parent, bg=self.bg_color)
        comp.pack(fill=tk.BOTH, expand=True, pady=5)
        
        pose_ctrl = tk.Frame(comp, bg=self.panel_bg, relief=tk.GROOVE, bd=2)
        pose_ctrl.pack(fill=tk.X, pady=(0, 8))
        
        pose_inner = tk.Frame(pose_ctrl, bg=self.panel_bg)
        pose_inner.pack(pady=10)
        
        tk.Label(
            pose_inner,
            text="Selected Pose:",
            font=("Cascadia Code", 11, "bold"),
            bg=self.panel_bg,
            fg="#dcddde"
        ).pack(side=tk.LEFT, padx=10)
        
        self.pose_var = tk.StringVar(value="Front Double Bicep")
        self.pose_combo = ttk.Combobox(
            pose_inner,
            textvariable=self.pose_var,
            state="readonly",
            font=("Cascadia Code", 10),
            width=25
        )
        self.pose_combo['values'] = list(self.pose_map.keys())
        self.pose_combo.pack(side=tk.LEFT, padx=5)
        self.pose_combo.bind("<<ComboboxSelected>>", self.on_pose_change)
        
        athletes = tk.Frame(comp, bg=self.bg_color)
        athletes.pack(fill=tk.BOTH, expand=True)
        athletes.grid_columnconfigure(0, weight=1)
        athletes.grid_columnconfigure(1, weight=1)
        athletes.grid_rowconfigure(0, weight=1)
        
        self.athlete1_name_label, self.lbl_img1 = self._athlete_panel(athletes, "Competitor 1", "#7f6df2", 0)
        self.athlete2_name_label, self.lbl_img2 = self._athlete_panel(athletes, "Competitor 2", "#a78bfa", 1)

    def _athlete_panel(self, parent, name, color, col):
        panel = tk.Frame(parent, bg=self.panel_bg, relief=tk.GROOVE, bd=2)
        panel.grid(row=0, column=col, sticky="nsew", padx=8, pady=5)
        
        name_lbl = tk.Label(
            panel,
            text=name,
            font=("Cascadia Code", 13, "bold"),
            bg=color,
            fg="#ffffff",
            pady=8
        )
        name_lbl.pack(fill=tk.X)
        
        img_area = tk.Frame(panel, bg="#1a1a1a", relief=tk.SUNKEN, bd=1)
        img_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        img_lbl = tk.Label(img_area, text="No Image Loaded", font=("Cascadia Code", 11), bg="#2d2d2d", fg="#808080")
        img_lbl.pack(expand=True)
        
        return name_lbl, img_lbl

    def _build_analysis_controls(self, parent):
        analysis = tk.Frame(parent, bg=self.panel_bg, relief=tk.GROOVE, bd=2)
        analysis.pack(fill=tk.X, pady=(8, 0))
        
        inner = tk.Frame(analysis, bg=self.panel_bg)
        inner.pack(pady=12, padx=10)
        
        tk.Label(inner, text="Analysis Options:", font=("Cascadia Code", 11, "bold"), bg=self.panel_bg, fg="#dcddde").pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            inner,
            text="X-Frame Analysis",
            command=self.run_xframe_analysis,
            font=("Cascadia Code", 10),
            bg=self.button_color,
            fg="white",
            relief=tk.RAISED,
            padx=12,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            inner,
            text="Conditioning Analysis",
            command=self.run_conditioning_analysis,
            font=("Cascadia Code", 10),
            bg=self.button_color,
            fg="white",
            relief=tk.RAISED,
            padx=12,
            pady=5
        ).pack(side=tk.LEFT, padx=5)

    def _build_log_panel(self, parent):
        log = tk.Frame(parent, bg=self.panel_bg, relief=tk.GROOVE, bd=2)
        log.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        
        hdr = tk.Frame(log, bg=self.accent_color)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Analysis Log", font=("Cascadia Code", 12, "bold"), bg=self.accent_color, fg="white", pady=10).pack()
        
        status = tk.Frame(log, bg=self.panel_bg)
        status.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(status, text="Status:", font=("Cascadia Code", 9, "bold"), bg=self.panel_bg, fg="#dcddde").pack(side=tk.LEFT)
        
        self.status_indicator = tk.Label(status, text="Ready", font=("Cascadia Code", 9), bg=self.panel_bg, fg=self.accent_color)
        self.status_indicator.pack(side=tk.LEFT, padx=5)
        
        txt_area = tk.Frame(log, bg=self.panel_bg)
        txt_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(txt_area)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            txt_area,
            font=("Courier", 9),
            bg="#1a1a1a",
            fg="#dcddde",
            relief=tk.SUNKEN,
            bd=1,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            padx=8,
            pady=8
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        tk.Button(log, text="Clear Log", command=self.clear_logs, font=("Cascadia Code", 9), bg="#3d3d3d", fg="#dcddde", relief=tk.RAISED, pady=5).pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self._log("System initialized", "info")

    def _truncate(self, path, max_len):
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len-3):]

    def _log(self, msg, level="info"):
        colors = {"info": "#dcddde", "success": "#8bc34a", "warning": "#ffa726", "error": "#f44336"}
        prefixes = {"info": "[INFO]", "success": "[OK]", "warning": "[WARN]", "error": "[ERROR]"}
        
        self.log_text.insert(tk.END, f"{prefixes[level]} ", "prefix")
        self.log_text.insert(tk.END, f"{msg}\n", level)
        self.log_text.tag_config(level, foreground=colors[level])
        self.log_text.tag_config("prefix", foreground=colors[level], font=("Courier", 9, "bold"))
        self.log_text.see(tk.END)

    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)
        self._log("Log cleared", "info")

    def _set_status(self, msg, level="info"):
        colors = {"info": self.accent_color, "success": "#8bc34a", "warning": "#ffa726", "error": "#f44336"}
        self.status_indicator.config(text=msg, fg=colors.get(level, self.accent_color))

    def browse_directory(self):
        directory = filedialog.askdirectory(title="Select Working Directory")
        if directory:
            self.working_directory = directory
            self.dir_label.config(text=self._truncate(directory, 60))
            self._log(f"Directory set: {directory}", "success")
            self._set_status("Directory updated", "success")

    def organize_poses(self):
        if not os.path.exists(self.working_directory):
            messagebox.showerror("Error", "Working directory does not exist!")
            self._log("Directory does not exist", "error")
            return
        
        self._log("Starting image organization...", "info")
        self._set_status("Organizing images...", "info")
        
        organizer = ImageOrganizer(self.working_directory)
        images = organizer.get_unorganized_images()
        
        if not images:
            messagebox.showinfo("Info", "No unorganized images found in directory.")
            self._log("No unorganized images found", "warning")
            self._set_status("Ready", "success")
            return
        
        self._log(f"Found {len(images)} unorganized images", "info")
        
        classified, unclassified = organizer.classify_images(images)
        
        if not classified:
            messagebox.showwarning("Warning", f"No poses could be identified from {len(images)} images.\nUnclassified files were NOT modified.")
            self._log("No poses could be identified", "warning")
            self._set_status("Ready", "success")
            return
        
        poses_dict = organizer.group_by_pose(classified)
        self._log(f"Identified poses: {', '.join(poses_dict.keys())}", "info")
        
        athlete1_poses, athlete2_poses = organizer.assign_to_athletes(poses_dict)
        existing_files = organizer.check_existing_files(athlete1_poses, athlete2_poses)
        
        if existing_files:
            response = messagebox.askyesno(
                "Overwrite Warning",
                f"The following organized files already exist:\n" + "\n".join(existing_files[:5]) + 
                ("\n..." if len(existing_files) > 5 else "") + f"\n\nDo you want to overwrite them?"
            )
            if not response:
                self._log("Organization cancelled by user", "warning")
                self._set_status("Cancelled", "warning")
                return
        
        renamed_count, errors = organizer.organize_files(athlete1_poses, athlete2_poses)
        
        athlete1_count = len(athlete1_poses)
        athlete2_count = len(athlete2_poses)
        
        self._log(f"Organized {renamed_count} images", "success")
        self._log(f"  Competitor 1: {athlete1_count} poses", "info")
        self._log(f"  Competitor 2: {athlete2_count} poses", "info")
        
        if unclassified:
            self._log(f"{len(unclassified)} images unclassified", "warning")
        if errors:
            self._log(f"{len(errors)} errors occurred", "error")
        
        msg = f"Organized {renamed_count} images:\n   Competitor 1: {athlete1_count} poses\n   Competitor 2: {athlete2_count} poses\n\n"
        msg += f"Poses found: {', '.join(poses_dict.keys())}\n"
        
        if unclassified:
            msg += f"\n{len(unclassified)} images could not be classified:\n   " + ", ".join(unclassified[:3])
            if len(unclassified) > 3:
                msg += f"... and {len(unclassified)-3} more"
            msg += "\n\n(These files were left unchanged)"
        
        if errors:
            msg += f"\nErrors: {len(errors)}"
        
        messagebox.showinfo("Organization Complete", msg)
        self._set_status("Organization complete", "success")

    def load_battle(self):
        self._log("Loading competitors...", "info")
        self._set_status("Loading comparison...", "info")
        
        loader = AthleteLoader(self.working_directory)
        athletes_dict = loader.load_athletes()
        self.athletes = athletes_dict
        athlete_names = loader.get_athlete_names(athletes_dict)
        
        if len(athlete_names) < 2:
            messagebox.showwarning("Warning", "Need at least 2 competitors organized in directory!")
            self._log("Not enough competitors found", "warning")
            self._set_status("Ready", "success")
            return
        
        self.current_athletes = athlete_names[:2]
        a1_display = athlete_names[0].replace('_', ' ').title()
        a2_display = athlete_names[1].replace('_', ' ').title()
        
        self.athlete1_name_label.config(text=a1_display)
        self.athlete2_name_label.config(text=a2_display)
        self.update_images()
        
        self._log(f"Comparison loaded: {a1_display} vs {a2_display}", "success")
        self._set_status("Comparison ready", "success")

    def on_pose_change(self, event):
        self.current_pose = self.pose_map[self.pose_var.get()]
        self.update_images()
        self._log(f"Pose changed to: {self.pose_var.get()}", "info")

    def update_images(self):
        if not self.current_athletes[0] or not self.current_athletes[1]:
            return
        
        if self.current_athletes[0] in self.athletes:
            if self.current_pose in self.athletes[self.current_athletes[0]]:
                path = self.athletes[self.current_athletes[0]][self.current_pose]
                self._show_image(path, self.lbl_img1)
            else:
                self.lbl_img1.config(image="", text=f"No {self.pose_var.get()} image")
        
        if self.current_athletes[1] in self.athletes:
            if self.current_pose in self.athletes[self.current_athletes[1]]:
                path = self.athletes[self.current_athletes[1]][self.current_pose]
                self._show_image(path, self.lbl_img2)
            else:
                self.lbl_img2.config(image="", text=f"No {self.pose_var.get()} image")

    def _show_image(self, filepath, label):
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                base_h = 380
                ratio = base_h / float(img.size[1])
                new_w = int(float(img.size[0]) * ratio)
                img = img.resize((new_w, base_h), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo, text="")
                label.image = photo
            except Exception as e:
                label.config(text=f"Error loading image", image="")
                self._log(f"Error loading image: {str(e)}", "error")
        else:
            label.config(image="", text=f"Image not found")

    def run_xframe_analysis(self):
        if not self.current_athletes[0] or not self.current_athletes[1]:
            messagebox.showwarning("Warning", "Please load a comparison first!")
            self._log("No comparison loaded for analysis", "warning")
            return
        
        self._log("Running X-Frame analysis...", "info")
        self._set_status("Analyzing X-Frame...", "info")
        
        results = {}
        annotated = {}
        
        for athlete in self.current_athletes:
            if athlete in self.athletes and self.current_pose in self.athletes[athlete]:
                img_path = self.athletes[athlete][self.current_pose]
                result = analyze_xframe(img_path)
                if result:
                    results[athlete] = result['metrics']
                    annotated[athlete] = result['annotated_image']
        
        if not results:
            messagebox.showerror("Error", "Could not analyze competitors. Check images.")
            self._log("X-Frame analysis failed", "error")
            self._set_status("Analysis failed", "error")
            return
        
        self._log("=" * 40, "info")
        self._log(f"X-FRAME ANALYSIS ({self.pose_var.get()})", "success")
        self._log("=" * 40, "info")
        
        for athlete, score in results.items():
            name = athlete.replace('_', ' ').title()
            self._log(f"\n{name}:", "info")
            self._log(f"  X-Frame: {score['x_frame']:.2f}", "info")
            self._log(f"  Mass: {score['mass']:.2f}", "info")
            self._log(f"  Shoulder: {score['shoulder']}px | Lat: {score['lat']}px", "info")
            self._log(f"  Waist: {score['waist']}px | Quad: {score['quad']}px", "info")
        
        if len(results) == 2:
            winner = max(results.keys(), key=lambda k: results[k]['x_frame'])
            self._log(f"\nHigher X-Frame Score: {winner.replace('_', ' ').title()}", "success")
        
        self._show_analysis(annotated, "X-Frame Analysis")
        self._set_status("X-Frame complete", "success")

    def run_conditioning_analysis(self):
        if not self.current_athletes[0] or not self.current_athletes[1]:
            messagebox.showwarning("Warning", "Please load a comparison first!")
            self._log("No comparison loaded for analysis", "warning")
            return
        
        self._log("Running Conditioning analysis...", "info")
        self._set_status("Analyzing conditioning...", "info")
        
        results = {}
        annotated = {}
        
        for athlete in self.current_athletes:
            if athlete in self.athletes and self.current_pose in self.athletes[athlete]:
                img_path = self.athletes[athlete][self.current_pose]
                
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    h, w = image.shape[:2]
                    config = AnalysisConfig()
                    pose_analyzer = PoseAnalyzer()
                    mask, landmarks = pose_analyzer.extract_mask_and_landmarks(image)
                    regions = RegionExtractor.get_analysis_regions(mask, landmarks, w, h)
                    
                    analysis_mask = regions.get('upper_back', regions['full_body'])
                    if np.sum(analysis_mask) < 1000:
                        analysis_mask = regions['full_body']
                    
                    analyzer = ConditioningAnalyzer(config)
                    metrics = analyzer.analyze(image, analysis_mask)
                    visualizer = ResultVisualizer(config)
                    annotated_img = visualizer.draw_overlay(image, metrics)
                    
                    results[athlete] = metrics
                    annotated[athlete] = annotated_img
                except Exception as e:
                    self._log(f"Error analyzing {athlete}: {str(e)}", "error")
                    continue
        
        if not results:
            messagebox.showerror("Error", "Could not analyze competitors. Check images.")
            self._log("Conditioning analysis failed", "error")
            self._set_status("Analysis failed", "error")
            return
        
        self._log("=" * 40, "info")
        self._log(f"CONDITIONING ANALYSIS ({self.pose_var.get()})", "success")
        self._log("=" * 40, "info")
        
        for athlete, metrics in results.items():
            name = athlete.replace('_', ' ').title()
            self._log(f"\n{name}:", "info")
            self._log(f"  Score: {metrics.overall_score:.1f}/100 ({metrics.grade})", "info")
            self._log(f"  Detail: {metrics.detail:.2f} | Separation: {metrics.separation:.2f}", "info")
            self._log(f"  Vascularity: {metrics.vascularity:.2f} | Tightness: {metrics.tightness:.2f}", "info")
        
        if len(results) == 2:
            winner = max(results.keys(), key=lambda k: results[k].overall_score)
            self._log(f"\nHigher Conditioning Score: {winner.replace('_', ' ').title()}", "success")
        
        self._show_analysis(annotated, "Conditioning Analysis")
        self._set_status("Conditioning complete", "success")

    def _show_analysis(self, annotated_imgs, title):
        if self.analysis_window and self.analysis_window.winfo_exists():
            self.analysis_window.destroy()
        
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title(f"{title} - Results")
        self.analysis_window.geometry("1400x700")
        
        self.analysis_window.update_idletasks()
        w = self.analysis_window.winfo_width()
        h = self.analysis_window.winfo_height()
        x = (self.analysis_window.winfo_screenwidth() // 2) - (w // 2)
        y = (self.analysis_window.winfo_screenheight() // 2) - (h // 2)
        self.analysis_window.geometry(f"{w}x{h}+{x}+{y}")
        
        container = tk.Frame(self.analysis_window, bg=self.bg_color)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)
        
        for i, (athlete, img_arr) in enumerate(annotated_imgs.items()):
            lbl_txt = athlete.replace('_', ' ').title()
            tk.Label(container, text=lbl_txt, font=("Cascadia Code", 13, "bold"), bg=self.panel_bg, fg="#dcddde", pady=10).grid(row=0, column=i, sticky="ew", padx=5, pady=5)
            
            img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            scale = 600 / h
            img_resized = cv2.resize(img_rgb, (int(w * scale), 600))
            pil_img = Image.fromarray(img_resized)
            photo = ImageTk.PhotoImage(pil_img)
            
            img_lbl = tk.Label(container, image=photo, bg=self.panel_bg)
            img_lbl.image = photo
            img_lbl.grid(row=1, column=i, sticky="nsew", padx=5, pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = BodybuildingCompetitionApp(root)
    root.mainloop()