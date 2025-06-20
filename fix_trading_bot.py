#!/usr/bin/env python3

# Leer el archivo y limpiar duplicados
with open('trading_bot.py', 'r') as f:
    content = f.read()

# Encontrar donde termina la clase
lines = content.split('\n')
cleaned_lines = []
in_duplicate = False

for line in lines:
    # Detectar inicio de método duplicado
    if 'def run_trading_cycle(self):' in line and len(cleaned_lines) > 10:
        in_duplicate = True
        continue
    
    # Si no estamos en duplicado, agregar línea
    if not in_duplicate:
        cleaned_lines.append(line)

# Agregar el método correcto y la propiedad symbol
cleaned_lines.append('')
cleaned_lines.append('    @property')
cleaned_lines.append('    def symbol(self):')
cleaned_lines.append('        """Default trading symbol"""')
cleaned_lines.append('        return getattr(self, "_symbol", "BTCUSDT")')
cleaned_lines.append('')
cleaned_lines.append('    @symbol.setter') 
cleaned_lines.append('    def symbol(self, value):')
cleaned_lines.append('        """Set trading symbol"""')
cleaned_lines.append('        self._symbol = value')
cleaned_lines.append('')
cleaned_lines.append('    def run_trading_cycle(self):')
cleaned_lines.append('        """Run a single trading cycle"""')
cleaned_lines.append('        try:')
cleaned_lines.append('            logger.info("📊 Running trading cycle...")')
cleaned_lines.append('            ')
cleaned_lines.append('            # Use symbol property')
cleaned_lines.append('            symbol = self.symbol')
cleaned_lines.append('            market_data = self.data_manager.get_market_data(symbol)')
cleaned_lines.append('            ')
cleaned_lines.append('            # Evaluate signals')
cleaned_lines.append('            signal = self.signal_evaluator.evaluate_signal(symbol, market_data)')
cleaned_lines.append('            ')
cleaned_lines.append('            # Log to database')
cleaned_lines.append('            if self.db_enabled and signal:')
cleaned_lines.append('                try:')
cleaned_lines.append('                    self.db_manager.log_signal(signal)')
cleaned_lines.append('                    logger.info("💾 Signal saved to database")')
cleaned_lines.append('                except Exception as e:')
cleaned_lines.append('                    logger.warning(f"Signal logging failed: {e}")')
cleaned_lines.append('            ')
cleaned_lines.append('            return {')
cleaned_lines.append('                "status": "completed",')
cleaned_lines.append('                "symbol": symbol,')
cleaned_lines.append('                "signal": signal,')
cleaned_lines.append('                "timestamp": datetime.now()')
cleaned_lines.append('            }')
cleaned_lines.append('            ')
cleaned_lines.append('        except Exception as e:')
cleaned_lines.append('            logger.error(f"Trading cycle failed: {e}")')
cleaned_lines.append('            return {')
cleaned_lines.append('                "status": "error",')
cleaned_lines.append('                "error": str(e),')
cleaned_lines.append('                "timestamp": datetime.now()')
cleaned_lines.append('            }')

# Escribir archivo limpio
with open('trading_bot.py', 'w') as f:
    f.write('\n'.join(cleaned_lines))

print("✅ trading_bot.py fixed")
