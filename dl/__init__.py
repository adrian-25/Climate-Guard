"""Research-only deep-learning comparison module for ClimateGuard.

The package is deliberately separate from the production Random Forest
prediction path. Importing it must never retrain, modify, or replace the
production model.
"""
