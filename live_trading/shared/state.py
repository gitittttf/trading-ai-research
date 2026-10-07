import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional

try:
    from live_trading.shared.config import DB_PATH
except ImportError:
    # Fallback if run standalone during dev/test
    DB_PATH = "data/trading_state.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # To easily convert to dict later
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self._create_tables()
            logger.info(f"Connected to StateManager DB at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_tables(self):
        try:
            with self.conn:
                # system_state
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS system_state (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                ''')

                # Default values for system_state
                self.conn.execute('''
                    INSERT OR IGNORE INTO system_state (key, value)
                    VALUES 
                        ('current_equity', '1000.0'),
                        ('breaker_penalty', '0.0')
                ''')

                # positions
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT,
                        trade_dir REAL,
                        entry_ts REAL,
                        entry_p1 REAL,
                        entry_p2 REAL,
                        entry_z REAL,
                        entry_std REAL,
                        delta_z_stop REAL,
                        adjusted_alpha REAL,
                        uncertainty REAL,
                        entry_adf REAL,
                        entry_volatility REAL,
                        z_momentum REAL,
                        v1 REAL,
                        v2 REAL,
                        entry_vol_usd REAL,
                        allocated_margin REAL,
                        size_p1 REAL,
                        size_p2 REAL,
                        meta_prob REAL,
                        exit_ts_preliminary REAL
                    )
                ''')

                # trades
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT,
                        trade_dir REAL,
                        entry_ts REAL,
                        entry_p1 REAL,
                        entry_p2 REAL,
                        entry_z REAL,
                        entry_std REAL,
                        delta_z_stop REAL,
                        adjusted_alpha REAL,
                        uncertainty REAL,
                        entry_adf REAL,
                        entry_volatility REAL,
                        z_momentum REAL,
                        v1 REAL,
                        v2 REAL,
                        entry_vol_usd REAL,
                        allocated_margin REAL,
                        size_p1 REAL,
                        size_p2 REAL,
                        meta_prob REAL,
                        exit_ts_preliminary REAL,
                        exit_ts REAL,
                        exit_p1 REAL,
                        exit_p2 REAL,
                        net_pnl REAL,
                        trade_cost REAL,
                        exit_reason TEXT,
                        rolling_breakdown_rate REAL,
                        balance_after REAL
                    )
                ''')
                # v2: full trade record as JSON (side, beta, legs, costs, ...)
                for table in ("positions", "trades"):
                    existing = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                    if "extra_json" not in existing:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN extra_json TEXT")
            logger.info("Tables checked/created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def _load_state(self) -> Dict[str, str]:
        try:
            cursor = self.conn.execute("SELECT key, value FROM system_state")
            return {row['key']: row['value'] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error loading state: {e}")
            raise

    def save_state(self, key: str, value: Any):
        try:
            with self.conn:
                self.conn.execute('''
                    INSERT INTO system_state (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                ''', (key, str(value)))
            logger.info(f"State saved: {key}={value}")
        except sqlite3.Error as e:
            logger.error(f"Error saving state for {key}: {e}")
            raise

    def add_open_position(self, trade_dict: Dict[str, Any]):
        cols = [
            'pair', 'trade_dir', 'entry_ts', 'entry_p1', 'entry_p2', 'v1', 'v2',
            'entry_z', 'entry_std', 'delta_z_stop', 'adjusted_alpha', 'uncertainty',
            'entry_adf', 'entry_volatility', 'z_momentum', 'entry_vol_usd',
            'allocated_margin', 'size_p1', 'size_p2', 'meta_prob', 'exit_ts_preliminary', 'extra_json'
        ]
        trade_dict = dict(trade_dict)
        trade_dict['extra_json'] = json.dumps(trade_dict.get('extra', {}), default=float)

        values = [trade_dict.get(col, 0.0 if col != 'pair' else None) for col in cols]
        
        placeholders = ', '.join(['?'] * len(cols))
        col_names = ', '.join(cols)
        
        try:
            with self.conn:
                self.conn.execute(f'''
                    INSERT INTO positions ({col_names})
                    VALUES ({placeholders})
                ''', values)
            logger.info(f"Added open position for {trade_dict.get('pair')}")
        except sqlite3.Error as e:
            logger.error(f"Error adding open position: {e}")
            raise

    def close_position(self, position_id: int, exit_data: Dict[str, Any]) -> None:
        try:
            with self.conn:
                # Fetch row
                cursor = self.conn.execute("SELECT * FROM positions WHERE id=?", (position_id,))
                pos = cursor.fetchone()
                
                if not pos:
                    logger.warning(f"Position with id {position_id} not found.")
                    return

                pos_dict = dict(pos)
                del pos_dict['id'] # remove id for insertion (we let AUTOINCREMENT handle new id, or we could insert it explicitly but prompt asked for AUTOINCREMENT on trades)

                # Merge with exit_data
                for k, v in exit_data.items():
                    pos_dict[k] = v

                # Ensure all trades columns are present in dict (defaults for missing)
                trades_cols = [
                    'pair', 'trade_dir', 'entry_ts', 'entry_p1', 'entry_p2', 'v1', 'v2',
                    'entry_z', 'entry_std', 'delta_z_stop', 'adjusted_alpha', 'uncertainty',
                    'entry_adf', 'entry_volatility', 'z_momentum', 'entry_vol_usd',
                    'allocated_margin', 'size_p1', 'size_p2', 'meta_prob', 'exit_ts_preliminary',
                    'exit_ts', 'exit_p1', 'exit_p2', 'net_pnl', 'trade_cost', 'exit_reason',
                    'rolling_breakdown_rate', 'balance_after', 'extra_json'
                ]
                if 'extra' in exit_data:
                    merged = json.loads(pos_dict.get('extra_json') or '{}')
                    merged.update(exit_data['extra'])
                    pos_dict['extra_json'] = json.dumps(merged, default=float)
                
                trade_values = [pos_dict.get(col, 0.0 if col not in ['pair', 'exit_reason', 'extra_json'] else None)
                                for col in trades_cols]
                placeholders = ', '.join(['?'] * len(trades_cols))
                col_names = ', '.join(trades_cols)
                
                # Insert into trades
                self.conn.execute(f'''
                    INSERT INTO trades ({col_names})
                    VALUES ({placeholders})
                ''', trade_values)
                
                # Delete from positions
                self.conn.execute("DELETE FROM positions WHERE id=?", (position_id,))
                
                # Update system_state
                new_equity = pos_dict.get('balance_after')
                if new_equity is not None:
                    self.conn.execute('''
                        INSERT INTO system_state (key, value)
                        VALUES (?, ?)
                        ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    ''', ('current_equity', str(new_equity)))
                
            logger.info(f"Closed position {position_id} and moved to trades.")
        except sqlite3.Error as e:
            logger.error(f"Error closing position {position_id}: {e}")
            raise

    def get_open_positions(self) -> List[Dict[str, Any]]:
        try:
            cursor = self.conn.execute("SELECT * FROM positions")
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting open positions: {e}")
            raise

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            cursor = self.conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error getting recent trades: {e}")
            raise

    def update_equity(self, new_equity: float):
        self.save_state('current_equity', new_equity)
        
    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
            logger.info("Database connection closed.")
        except sqlite3.Error as e:
            logger.error(f"Error closing database: {e}")
            raise
