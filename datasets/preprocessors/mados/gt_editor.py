"""
datasets/preprocessors/mados/gt_editor.py
------------------------------------------
Editor de Ground Truth para MADOS - v3 (tkinter)

Controles:
    Clic izquierdo     -> pintar clase activa
    Clic derecho       -> borrar
    Rueda raton        -> zoom in/out
    Ctrl + Z           -> deshacer
    S                  -> guardar y siguiente
    X                  -> saltar
    Q                  -> salir
    A                  -> Aplicar FAI a toda la imagen

Uso:
    python -m datasets.preprocessors.mados.gt_editor --split train --solo-sargassum
    python -m datasets.preprocessors.mados.gt_editor --split train
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from core.config.paths import SARGASSUM_READY

TARGET = 224
ZOOM_LEVELS = [1, 2, 3, 4, 6, 8, 10, 12]

CLASE_COLORES_RGBA = {
    0: (0,   0,   0,   0),
    2: (20,  200, 50,  180),
    3: (120, 255, 120, 180),
}
CLASE_NOMBRES = {0: "Borrar", 2: "Dense Sargassum", 3: "Sparse Algae"}


def _preparar_rgb(img_raw: np.ndarray) -> np.ndarray:
    img = img_raw.copy().astype(np.float32)
    if img.max() > 10.0:
        img /= 10000.0
    h, w = img.shape[:2]
    y0, x0 = (h - TARGET) // 2, (w - TARGET) // 2
    img = img[y0:y0+TARGET, x0:x0+TARGET, :]
    img = img[:, :, [2, 1, 0]]          # (B,G,R,NIR) -> (R,G,B)
    img = np.clip(img * 5.0, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _preparar_fai(img_raw: np.ndarray) -> np.ndarray:
    img = img_raw.copy().astype(np.float32)
    if img.max() > 10.0:
        img /= 10000.0
    h, w = img.shape[:2]
    y0, x0 = (h - TARGET) // 2, (w - TARGET) // 2
    img = img[y0:y0+TARGET, x0:x0+TARGET, :]
    fai = img[:, :, 3] - img[:, :, 2]   # NIR - Rojo
    # Normalizar a RGB para mostrar: verde=positivo, rojo=negativo
    vmax = max(abs(fai).max(), 0.02)
    norm = np.clip(fai / vmax, -1.0, 1.0)
    r = np.clip(-norm, 0, 1)
    g = np.clip( norm, 0, 1)
    b = np.zeros_like(norm)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    y0, x0 = (h - TARGET) // 2, (w - TARGET) // 2
    return mask[y0:y0+TARGET, x0:x0+TARGET]


def _mask_a_rgba(mask_crop: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask_crop.shape, 4), dtype=np.uint8)
    for clase, color in CLASE_COLORES_RGBA.items():
        rgba[mask_crop == clase] = color
    return rgba


def _ordenar_tiles(img_dir: Path, mask_dir: Path,
                   solo_sargassum: bool) -> list[Path]:
    todos = sorted(img_dir.glob("*.npy"))
    con_sarg, sin_sarg = [], []
    for p in todos:
        mp = mask_dir / p.name
        if not mp.exists():
            continue
        m = np.load(mp)
        if np.isin(m, [2, 3]).any():
            con_sarg.append(p)
        elif not solo_sargassum:
            sin_sarg.append(p)
    print(f"Con sargazo: {len(con_sarg)}  Sin sargazo: {len(sin_sarg)}")
    return con_sarg + sin_sarg


class GTEditorApp:

    def __init__(self, root: tk.Tk, tiles: list[Path],
                 mask_dir: Path, backup_dir: Path) -> None:
        self.root       = root
        self.tiles      = tiles
        self.mask_dir   = mask_dir
        self.backup_dir = backup_dir

        self.idx        = 0
        self.guardados  = 0
        self.saltados   = 0

        self.clase_activa = tk.IntVar(value=2)
        self.zoom_idx     = 2           # zoom level index -> ZOOM_LEVELS[2] = 3
        self.historial    = []
        self.mask_edit    = None
        self.mask_original = None
        self.img_rgb      = None
        self.img_fai      = None
        self._pan_x       = 0
        self._pan_y       = 0
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._paneando    = False

        self._build_ui()
        self._cargar_tile()

    # ─────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("GT Editor — MADOS")
        self.root.configure(bg="#111827")
        self.root.bind("<Key>", self._on_key)

        # Barra superior
        top = tk.Frame(self.root, bg="#1f2937", pady=4)
        top.pack(fill="x")

        self.lbl_titulo = tk.Label(
            top, text="", bg="#1f2937", fg="white", font=("Courier", 10, "bold")
        )
        self.lbl_titulo.pack(side="left", padx=10)

        tk.Label(top, text="S=guardar  X=saltar  Ctrl+Z=deshacer  A=Aplicar FAI  R=reset  Q=salir  |  Rueda=zoom  Espacio+drag=mover",
                 bg="#1f2937", fg="#9ca3af", font=("Courier", 8)).pack(side="right", padx=10)

        # Cuerpo principal
        body = tk.Frame(self.root, bg="#111827")
        body.pack(fill="both", expand=True)

        # Panel izquierdo: referencias
        left = tk.Frame(body, bg="#111827", width=250)
        left.pack(side="left", fill="y", padx=4, pady=4)
        left.pack_propagate(True) # Para que no corte el contenido verticalmente

        tk.Label(left, text="RGB", bg="#111827", fg="#9ca3af",
                 font=("Courier", 8)).pack()
        self.lbl_rgb = tk.Label(left, bg="#000")
        self.lbl_rgb.pack(pady=2)

        tk.Label(left, text="FAI  (verde=sargazo)", bg="#111827", fg="#9ca3af",
                 font=("Courier", 8)).pack()
        self.lbl_fai = tk.Label(left, bg="#000")
        self.lbl_fai.pack(pady=2)

        tk.Label(left, text="GT original", bg="#111827", fg="#9ca3af",
                 font=("Courier", 8)).pack()
        self.lbl_orig = tk.Label(left, bg="#000")
        self.lbl_orig.pack(pady=2)

        # Clase activa
        tk.Label(left, text="─── Clase activa ───", bg="#111827",
                 fg="#4ade80", font=("Courier", 8, "bold")).pack(pady=(8, 2))
        for k, nombre in CLASE_NOMBRES.items():
            color = {0: "#ef4444", 2: "#22c55e", 3: "#86efac"}[k]
            tk.Radiobutton(
                left, text=nombre, variable=self.clase_activa, value=k,
                bg="#111827", fg=color, selectcolor="#374151",
                activebackground="#111827", activeforeground=color,
                font=("Courier", 9, "bold"),
            ).pack(anchor="w", padx=8)

        tk.Label(left, text="─── Info ───", bg="#111827",
                 fg="#9ca3af", font=("Courier", 8)).pack(pady=(10, 2))
        self.lbl_stats = tk.Label(
            left, text="", bg="#111827", fg="white", font=("Courier", 9),
            justify="left",
        )
        self.lbl_stats.pack(anchor="w", padx=8)

        # Botones
        for (txt, cmd, col) in [
            ("S  Guardar →",  self._guardar,  "#166534"),
            ("X  Saltar  →",  self._saltar,   "#374151"),
            ("Z  Deshacer",   self._deshacer, "#1e3a5f"),
        ]:
            tk.Button(
                left, text=txt, command=cmd,
                bg=col, fg="white", font=("Courier", 9, "bold"),
                relief="flat", cursor="hand2", pady=4,
            ).pack(fill="x", padx=6, pady=2)

        # Auto-FAI (Botones en lugar de Slider para evitar bugs gráficos)
        tk.Label(left, text="─── Pintar por FAI ───", bg="#111827",
                 fg="#f59e0b", font=("Courier", 8, "bold")).pack(pady=(10, 2))

        fai_frame = tk.Frame(left, bg="#111827")
        fai_frame.pack(fill="x", padx=6)
        
        tk.Label(fai_frame, text="Umbral:", bg="#111827",
                 fg="#9ca3af", font=("Courier", 8)).pack(side="left")
        
        self.fai_umbral = tk.DoubleVar(value=0.005) 
        
        # Botón Menos
        tk.Button(fai_frame, text="-", command=lambda: self._cambiar_umbral(-0.001),
                  bg="#374151", fg="white", font=("Courier", 8, "bold"),
                  relief="flat", cursor="hand2", padx=4).pack(side="left", padx=2)
                  
        # Etiqueta del valor
        self.lbl_fai_val = tk.Label(fai_frame, text="0.005", bg="#111827", width=6,
                                     fg="#f59e0b", font=("Courier", 8, "bold"))
        self.lbl_fai_val.pack(side="left", padx=2)
        
        # Botón Más
        tk.Button(fai_frame, text="+", command=lambda: self._cambiar_umbral(0.001),
                  bg="#374151", fg="white", font=("Courier", 8, "bold"),
                  relief="flat", cursor="hand2", padx=4).pack(side="left", padx=2)


        fai_clase_frame = tk.Frame(left, bg="#111827")
        fai_clase_frame.pack(fill="x", padx=6, pady=6)
        tk.Label(fai_clase_frame, text="Como:", bg="#111827",
                 fg="#9ca3af", font=("Courier", 8)).pack(side="left")
        self.fai_clase = tk.IntVar(value=3) # Por defecto expande como Sparse Algae
        for k, lbl, col in [(2, "Densa", "#22c55e"), (3, "Escasa", "#86efac")]:
            tk.Radiobutton(
                fai_clase_frame, text=lbl, variable=self.fai_clase, value=k,
                bg="#111827", fg=col, selectcolor="#374151",
                activebackground="#111827", font=("Courier", 8),
            ).pack(side="left", padx=2)

        tk.Button(
            left, text="A  Aplicar FAI",
            command=self._aplicar_fai,
            bg="#92400e", fg="white", font=("Courier", 9, "bold"),
            relief="flat", cursor="hand2", pady=4,
        ).pack(fill="x", padx=6, pady=2)

        tk.Button(
            left, text="Limpiar todo",
            command=self._limpiar_todo,
            bg="#450a0a", fg="#fca5a5", font=("Courier", 8),
            relief="flat", cursor="hand2", pady=3,
        ).pack(fill="x", padx=6, pady=1)

        # Canvas central de edicion
        center = tk.Frame(body, bg="#111827")
        center.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        tk.Label(center, text="EDICION  (clic izq=pintar  clic der=borrar)",
                 bg="#111827", fg="white", font=("Courier", 9, "bold")).pack()

        self.canvas = tk.Canvas(center, bg="#000", cursor="crosshair",
                                highlightthickness=1, highlightbackground="#4ade80")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>",        self._on_press_r)
        self.canvas.bind("<B3-Motion>",       self._on_drag_r)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<MouseWheel>",      self._on_wheel)
        self.canvas.bind("<Button-4>",        self._on_wheel)   # Linux scroll up
        self.canvas.bind("<Button-5>",        self._on_wheel)   # Linux scroll down
        self.canvas.bind("<Configure>",       lambda e: self._redraw())
        # Pan con clic del medio
        self.canvas.bind("<Button-2>",        self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        # Pan con barra espaciadora + arrastre
        self.root.bind("<space>",             self._on_space_press)
        self.root.bind("<KeyRelease-space>",  self._on_space_release)

        self._pintando = False
        self._borrando = False
        self._guardado_historial_este_trazo = False

    def _cambiar_umbral(self, delta: float) -> None:
        nuevo_valor = self.fai_umbral.get() + delta
        # Limitamos entre -0.02 y 0.05
        nuevo_valor = max(-0.02, min(0.05, nuevo_valor))
        self.fai_umbral.set(nuevo_valor)
        self.lbl_fai_val.config(text=f"{nuevo_valor:.3f}")

    # ─────────────────────────────────────────────────────────────────
    # CARGAR TILE
    # ─────────────────────────────────────────────────────────────────

    def _cargar_tile(self) -> None:
        if self.idx >= len(self.tiles):
            self._fin()
            return

        img_path       = self.tiles[self.idx]
        mask_path      = self.mask_dir / img_path.name
        img_raw        = np.load(img_path).astype(np.float32)
        self.mask_original = np.load(mask_path).astype(np.uint8)
        self.mask_edit     = self.mask_original.copy()
        self.historial     = []
        self._nombre       = img_path.name
        self._mask_path    = mask_path
        self._pan_x        = 0
        self._pan_y        = 0

        self.img_rgb  = _preparar_rgb(img_raw)
        self.img_fai  = _preparar_fai(img_raw)
        self.fai_raw  = self._calcular_fai_raw(img_raw)  # valores numericos para auto-fill

        n_tot = len(self.tiles)
        self.lbl_titulo.config(
            text=f"[{self.idx+1}/{n_tot}]  {img_path.name}"
        )

        # Paneles pequeños (112x112)
        self._ph_rgb  = ImageTk.PhotoImage(
            Image.fromarray(self.img_rgb).resize((112, 112), Image.NEAREST)
        )
        self._ph_fai  = ImageTk.PhotoImage(
            Image.fromarray(self.img_fai).resize((112, 112), Image.NEAREST)
        )
        self.lbl_rgb.config(image=self._ph_rgb)
        self.lbl_fai.config(image=self._ph_fai)

        self._actualizar_orig()
        self._actualizar_stats()
        self._redraw()

    def _actualizar_orig(self) -> None:
        mc = _crop_mask(self.mask_original)
        rgba_orig = _mask_a_rgba(mc)
        base = Image.fromarray(self.img_rgb)
        overlay = Image.fromarray(rgba_orig, mode="RGBA")
        comp = base.convert("RGBA")
        comp.paste(overlay, mask=overlay.split()[3])
        self._ph_orig = ImageTk.PhotoImage(
            comp.resize((112, 112), Image.NEAREST)
        )
        self.lbl_orig.config(image=self._ph_orig)

    # ─────────────────────────────────────────────────────────────────
    # DIBUJO EN CANVAS
    # ─────────────────────────────────────────────────────────────────

    def _zoom(self) -> int:
        return ZOOM_LEVELS[self.zoom_idx]

    def _redraw(self) -> None:
        z = self._zoom()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        mc = _crop_mask(self.mask_edit)
        rgba = _mask_a_rgba(mc)

        base    = Image.fromarray(self.img_rgb).convert("RGBA")
        overlay = Image.fromarray(rgba, mode="RGBA")
        comp    = base.copy()
        comp.paste(overlay, mask=overlay.split()[3])

        w_zoom = TARGET * z
        h_zoom = TARGET * z
        img_zoom = comp.resize((w_zoom, h_zoom), Image.NEAREST)

        # Offset base (centrado) + pan del usuario
        base_x = max(0, (cw - w_zoom) // 2)
        base_y = max(0, (ch - h_zoom) // 2)
        self._off_x = base_x + self._pan_x
        self._off_y = base_y + self._pan_y

        self._ph_edit = ImageTk.PhotoImage(img_zoom)
        self.canvas.delete("all")
        self.canvas.create_image(self._off_x, self._off_y,
                                  anchor="nw", image=self._ph_edit)

    def _canvas_a_pixel(self, cx: int, cy: int) -> tuple[int, int]:
        """Convierte coordenadas de canvas a coordenadas de pixel de la imagen."""
        z  = self._zoom()
        px = (cx - self._off_x) // z
        py = (cy - self._off_y) // z
        return px, py

    def _pintar_pixel(self, cx: int, cy: int, clase: int) -> None:
        px, py = self._canvas_a_pixel(cx, cy)
        h, w   = self.mask_edit.shape
        TARGET_h, TARGET_w = TARGET, TARGET
        mh, mw = self.mask_edit.shape
        y0_crop = (mh - TARGET_h) // 2
        x0_crop = (mw - TARGET_w) // 2

        # Convertir de coordenadas de crop a coordenadas de mascara completa
        my = py + y0_crop
        mx = px + x0_crop

        if 0 <= my < h and 0 <= mx < w:
            if self.mask_edit[my, mx] != clase:
                self.mask_edit[my, mx] = clase
                self._redraw()
                self._actualizar_stats()

    # ─────────────────────────────────────────────────────────────────
    # EVENTOS
    # ─────────────────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if self._paneando:
            self._pan_start_x = event.x - self._pan_x
            self._pan_start_y = event.y - self._pan_y
            return
        self._pintando = True
        if not self._guardado_historial_este_trazo:
            self._guardar_historial()
            self._guardado_historial_este_trazo = True
        self._pintar_pixel(event.x, event.y, self.clase_activa.get())

    def _on_drag(self, event) -> None:
        if self._paneando:
            self._pan_x = event.x - self._pan_start_x
            self._pan_y = event.y - self._pan_start_y
            self._redraw()
            return
        if self._pintando:
            self._pintar_pixel(event.x, event.y, self.clase_activa.get())

    def _on_press_r(self, event) -> None:
        self._borrando = True
        if not self._guardado_historial_este_trazo:
            self._guardar_historial()
            self._guardado_historial_este_trazo = True
        self._pintar_pixel(event.x, event.y, 0)

    def _on_drag_r(self, event) -> None:
        if self._borrando:
            self._pintar_pixel(event.x, event.y, 0)

    def _on_release(self, event) -> None:
        self._pintando = False
        self._borrando = False
        self._guardado_historial_este_trazo = False

    def _on_wheel(self, event) -> None:
        # Windows: event.delta  /  Linux: event.num
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self.zoom_idx = min(len(ZOOM_LEVELS) - 1, self.zoom_idx + 1)
        else:
            self.zoom_idx = max(0, self.zoom_idx - 1)
        self._redraw()

    def _on_pan_start(self, event) -> None:
        self._pan_start_x = event.x - self._pan_x
        self._pan_start_y = event.y - self._pan_y
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event) -> None:
        self._pan_x = event.x - self._pan_start_x
        self._pan_y = event.y - self._pan_start_y
        self._redraw()

    def _on_pan_end(self, event) -> None:
        self.canvas.config(cursor="crosshair")

    def _on_space_press(self, event) -> None:
        if not self._paneando:
            self._paneando = True
            self.canvas.config(cursor="fleur")

    def _on_space_release(self, event) -> None:
        self._paneando = False
        self.canvas.config(cursor="crosshair")

    def _on_key(self, event) -> None:
        k = event.keysym.lower()
        if k == "s":
            self._guardar()
        elif k == "x":
            self._saltar()
        elif k == "z" and event.state & 0x4:   # Ctrl+Z
            self._deshacer()
        elif k == "a":
            self._aplicar_fai()
        elif k == "r":
            self._pan_x = 0
            self._pan_y = 0
            self._redraw()
        elif k == "q":
            self._fin()

    # ─────────────────────────────────────────────────────────────────
    # ACCIONES
    # ─────────────────────────────────────────────────────────────────

    def _guardar(self) -> None:
        bk = self.backup_dir / self._mask_path.name
        if not bk.exists():
            np.save(bk, self.mask_original)
        np.save(self._mask_path, self.mask_edit)
        self.guardados += 1
        n_n = int(np.isin(self.mask_edit, [2, 3]).sum())
        n_o = int(np.isin(self.mask_original, [2, 3]).sum())
        d   = n_n - n_o
        print(f"  OK {self._nombre}  {n_o}px -> {n_n}px  ({'+' if d>=0 else ''}{d}px)")
        self.idx += 1
        self._cargar_tile()

    def _saltar(self) -> None:
        self.saltados += 1
        self.idx += 1
        self._cargar_tile()

    def _deshacer(self) -> None:
        if self.historial:
            self.mask_edit = self.historial.pop()
            self._redraw()
            self._actualizar_stats()

    def _guardar_historial(self) -> None:
        self.historial.append(self.mask_edit.copy())
        if len(self.historial) > 50:
            self.historial.pop(0)

    def _calcular_fai_raw(self, img_raw: np.ndarray) -> np.ndarray:
        """Devuelve el mapa FAI numerico (224x224) para uso en auto-fill."""
        img = img_raw.copy().astype(np.float32)
        if img.max() > 10.0:
            img /= 10000.0
        h, w = img.shape[:2]
        y0, x0 = (h - TARGET) // 2, (w - TARGET) // 2
        img = img[y0:y0+TARGET, x0:x0+TARGET, :]
        return img[:, :, 3] - img[:, :, 2]   # NIR - Rojo (orden B,G,R,NIR)

    def _aplicar_fai(self) -> None:
        """Pinta automaticamente todos los pixeles donde FAI >= umbral."""
        umbral = self.fai_umbral.get()
        clase  = self.fai_clase.get()
        self._guardar_historial()

        mh, mw = self.mask_edit.shape
        y0_crop = (mh - TARGET) // 2
        x0_crop = (mw - TARGET) // 2

        mascara_fai = self.fai_raw >= umbral   # (224, 224) bool

        # Aplicar al area de crop de la mascara completa
        region = self.mask_edit[y0_crop:y0_crop+TARGET, x0_crop:x0_crop+TARGET]
        region[mascara_fai] = clase
        self.mask_edit[y0_crop:y0_crop+TARGET, x0_crop:x0_crop+TARGET] = region

        n = int(mascara_fai.sum())
        print(f"  [Auto-FAI] umbral={umbral:.3f} clase={clase} → {n}px pintados")
        self._redraw()
        self._actualizar_stats()

    def _limpiar_todo(self) -> None:
        """Borra todas las etiquetas de sargazo del tile actual."""
        self._guardar_historial()
        mh, mw = self.mask_edit.shape
        y0_crop = (mh - TARGET) // 2
        x0_crop = (mw - TARGET) // 2
        region = self.mask_edit[y0_crop:y0_crop+TARGET, x0_crop:x0_crop+TARGET]
        region[np.isin(region, [2, 3])] = 0
        self.mask_edit[y0_crop:y0_crop+TARGET, x0_crop:x0_crop+TARGET] = region
        self._redraw()
        self._actualizar_stats()

    def _fin(self) -> None:
        print(f"\n[GT Editor] Fin  guardados={self.guardados} saltados={self.saltados}")
        self.root.destroy()

    # ─────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────

    def _actualizar_stats(self) -> None:
        n2 = int((self.mask_edit == 2).sum())
        n3 = int((self.mask_edit == 3).sum())
        self.lbl_stats.config(
            text=f"Dense Sarg.:\n  {n2} px\n\nSparse Algae:\n  {n3} px\n\nTotal:\n  {n2+n3} px"
        )


# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",          default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--dataset",        default=str(SARGASSUM_READY))
    parser.add_argument("--solo-sargassum", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    img_dir     = dataset_dir / args.split / "images"
    mask_dir    = dataset_dir / args.split / "masks"
    backup_dir  = dataset_dir / args.split / "masks_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    tiles = _ordenar_tiles(img_dir, mask_dir, args.solo_sargassum)
    if not tiles:
        print(f"[GT Editor] Sin tiles en {img_dir}")
        sys.exit(0)

    print(f"[GT Editor] {len(tiles)} tiles · split='{args.split}'")

    root = tk.Tk()
    root.geometry("1400x860")
    app  = GTEditorApp(root, tiles, mask_dir, backup_dir)
    root.mainloop()


if __name__ == "__main__":
    main()