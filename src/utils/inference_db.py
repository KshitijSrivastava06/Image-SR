"""
Inference Tracker — SQLite database for tracking super-resolution runs and benchmarks.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.common import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("inference_db")


class InferenceTracker:
    def __init__(self, db_path: str = "outputs/inference_runs.db"):
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    model_arch TEXT NOT NULL,
                    scale_factor INTEGER NOT NULL,
                    
                    device TEXT,
                    inference_time_ms REAL,
                    vram_peak_mb REAL,
                    
                    input_filename TEXT,
                    input_width INTEGER,
                    input_height INTEGER,
                    
                    dataset_name TEXT,
                    psnr REAL,
                    ssim REAL
                )
            ''')
            conn.commit()

    def log_inference(
        self,
        model_arch: str,
        scale_factor: int,
        device: str,
        inference_time_ms: float,
        vram_peak_mb: Optional[float] = None,
        input_filename: Optional[str] = None,
        input_width: Optional[int] = None,
        input_height: Optional[int] = None,
        dataset_name: Optional[str] = None,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
    ) -> int:
        """Log a single inference or benchmark run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO inference_runs (
                    model_arch, scale_factor, device, inference_time_ms, vram_peak_mb,
                    input_filename, input_width, input_height, dataset_name, psnr, ssim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_arch, scale_factor, device, inference_time_ms, vram_peak_mb,
                input_filename, input_width, input_height, dataset_name, psnr, ssim
            ))
            conn.commit()
            return cursor.lastrowid

    def get_history(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent runs."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM inference_runs ORDER BY timestamp DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_benchmark_stats(self) -> pd.DataFrame:
        """Aggregate PSNR/SSIM grouped by model_arch and dataset_name."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    model_arch, 
                    dataset_name,
                    COUNT(*) as num_runs,
                    AVG(psnr) as avg_psnr,
                    AVG(ssim) as avg_ssim,
                    AVG(inference_time_ms) as avg_inference_time_ms
                FROM inference_runs
                WHERE dataset_name IS NOT NULL AND psnr IS NOT NULL
                GROUP BY model_arch, dataset_name
            '''
            return pd.read_sql_query(query, conn)
