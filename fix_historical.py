#!/usr/bin/env python3

# Leer el archivo y reemplazar el método problemático
with open('historical_data_downloader.py', 'r') as f:
    content = f.read()

# Reemplazar el método _save_session_to_db para que use menos parámetros
new_method = '''    async def _save_session_to_db(self, session) -> bool:
        """Save download session to database (simplified)"""
        try:
            if not self.db_manager:
                return False
            
            await self.db_manager.log_download_session(
                session_id=session.session_id,
                symbol=session.symbol,
                start_date=session.start_date,
                end_date=session.end_date,
                timeframe=session.timeframe,
                status=session.status,
                records_processed=session.records_processed,
                total_records_expected=session.total_records_expected,
                progress_percentage=session.progress_percentage
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session to database: {e}")
            return False'''

# Buscar y reemplazar el método existente
import re
pattern = r'async def _save_session_to_db\(self, session.*?return False'
content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# Escribir de vuelta
with open('historical_data_downloader.py', 'w') as f:
    f.write(content)

print("✅ Historical downloader fixed")
