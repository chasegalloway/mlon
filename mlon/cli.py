"""
Command-line interface for MLON's automated checks.

This module provides a production-grade CLI for running automated ML safety checks.
It includes progress tracking, error handling, and detailed logging.
"""
import click
import pandas as pd
import sys
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import logging

from .auto_check import AutoChecker
from .logging import setup_logging, get_logger
from .exceptions import MLONError, MLONDataError

logger = get_logger(__name__)

def setup_cli_logging(verbose: bool) -> None:
    """Configure logging for CLI."""
    log_dir = Path.home() / ".mlon" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"mlon_{datetime.now().strftime('%Y%m%d')}.log"
    
    setup_logging(
        log_level="DEBUG" if verbose else "INFO",
        log_file=str(log_file)
    )

@click.group()
def cli():
    """MLON CLI - One-line ML safety checks"""
    pass

@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--train-data', type=click.Path(exists=True), help='Path to training data (optional)')
@click.option('--test-data', type=click.Path(exists=True), help='Path to test data (optional)')
@click.option('--target', type=str, help='Target column name (will be inferred if not provided)')
@click.option('--protected', type=str, multiple=True, help='Protected attribute columns (can be specified multiple times)')
@click.option('--output', type=str, default='mlon_report.pdf', help='Output path for PDF report')
def check(data_path: str,
         train_data: Optional[str] = None,
         test_data: Optional[str] = None,
         target: Optional[str] = None,
         protected: Optional[tuple] = None,
         output: str = 'mlon_report.pdf'):
    """
    Run automated ML safety checks on your data.
    
    Example usage:
    
    \b
    # Basic check
    mlon check data.csv
    
    \b
    # Check with train/test split
    mlon check data.csv --train-data train.csv --test-data test.csv
    
    \b
    # Specify target and protected attributes
    mlon check data.csv --target outcome --protected gender --protected race
    """
    # Load data
    data = pd.read_csv(data_path)
    train_df = pd.read_csv(train_data) if train_data else None
    test_df = pd.read_csv(test_data) if test_data else None
    
    # Initialize checker
    checker = AutoChecker()
    
    # Run checks
    click.echo("🔍 Running ML safety checks...")
    results = checker.check_data(
        data=data,
        train_data=train_df,
        test_data=test_df,
        target_col=target,
        protected_features=list(protected) if protected else None,
        output_path=output
    )
    
    # Print summary to console
    click.echo("\n✨ Check completed! Summary of findings:")
    
    if 'data_overview' in results:
        click.echo("\n📊 Data Overview:")
        for key, value in results['data_overview'].items():
            click.echo(f"  • {key.replace('_', ' ').title()}: {value}")
    
    if 'leakage_detection' in results:
        click.echo("\n🔍 Data Leakage:")
        for check_type, warnings in results['leakage_detection'].items():
            if warnings:
                click.echo(f"\n  ⚠️  {check_type.replace('_', ' ').title()}:")
                for warning in warnings:
                    click.echo(f"    • {warning}")
            else:
                click.echo(f"\n  ✅ {check_type.replace('_', ' ').title()}: No issues detected")
    
    if 'bias_detection' in results:
        click.echo("\n⚖️  Bias Detection:")
        for feature, warnings in results['bias_detection'].items():
            if any(len(w) > 0 for w in warnings.values()):
                click.echo(f"\n  ⚠️  {feature}:")
                for warning_type, warning_list in warnings.items():
                    if warning_list:
                        for warning in warning_list:
                            click.echo(f"    • {warning}")
            else:
                click.echo(f"\n  ✅ {feature}: No issues detected")
    
    if 'recommendations' in results:
        click.echo("\n💡 Recommendations:")
        for rec in results['recommendations']:
            click.echo(f"  • {rec}")
    
    click.echo(f"\n📄 Detailed report saved to: {output}")

if __name__ == '__main__':
    cli()
