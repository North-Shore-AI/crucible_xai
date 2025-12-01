# Suppress info/debug logs during tests - XAI methods are verbose
Logger.configure(level: :warning)

ExUnit.start()
