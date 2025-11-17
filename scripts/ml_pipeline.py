#!/usr/bin/env python3
"""
ML Pipeline Orchestrator for NYC 311 Data
Coordinates all ML tasks: Classification, Forecasting, and Regression
"""

import os
import sys
import subprocess
from datetime import datetime

class MLPipelineOrchestrator:
    def __init__(self):
        self.results = {}
        
    def run_ml_task(self, script_name, description):
        """Run an ML task and handle errors"""
        print(f"\n{'='*70}")
        print(f"🚀 Starting: {description}")
        print(f"{'='*70}")
        print(f"📝 Running: {script_name}")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run([
                sys.executable, f"/app/scripts/{script_name}"
            ], capture_output=True, text=True, cwd="/app")
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                print(f"\n✅ COMPLETED: {description}")
                print(f"⏱️  Duration: {duration:.2f} seconds")
                
                # Show last 50 lines of output
                output_lines = result.stdout.split('\n')
                if len(output_lines) > 50:
                    print("\n📊 Output (last 50 lines):")
                    print('\n'.join(output_lines[-50:]))
                else:
                    print("\n📊 Output:")
                    print(result.stdout)
                
                self.results[script_name] = {
                    'status': 'success',
                    'duration': duration
                }
                return True
            else:
                print(f"\n❌ FAILED: {description}")
                print(f"⏱️  Duration: {duration:.2f} seconds")
                print("\n📋 Error Output:")
                print(result.stderr)
                
                self.results[script_name] = {
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr
                }
                return False
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n❌ EXCEPTION in {description}: {e}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            
            self.results[script_name] = {
                'status': 'exception',
                'duration': duration,
                'error': str(e)
            }
            return False

    def print_summary(self):
        """Print pipeline execution summary"""
        print(f"\n{'='*70}")
        print("📊 ML PIPELINE EXECUTION SUMMARY")
        print(f"{'='*70}")
        
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed_tasks = total_tasks - successful_tasks
        total_duration = sum(r['duration'] for r in self.results.values())
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   ✅ Successful: {successful_tasks}")
        print(f"   ❌ Failed: {failed_tasks}")
        print(f"   ⏱️  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        print(f"\n📋 Task Details:")
        for script_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"   {status_icon} {script_name}: {result['status']} ({result['duration']:.2f}s)")
        
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        return successful_tasks == total_tasks

def main():
    print("="*70)
    print("🤖 NYC 311 ML PIPELINE ORCHESTRATOR")
    print("="*70)
    print(f"🕐 Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    
    orchestrator = MLPipelineOrchestrator()
    
    # Define ML tasks to run
    tasks = [
        ("ml_classification.py", "Multiclass Classification (Logistic Regression, Random Forest, XGBoost)"),
        ("ml_forecasting.py", "Time Series Forecasting (Prophet)"),
        ("spark_regression.py", "Spark MLlib Linear Regression"),
    ]
    
    # Run all tasks
    for script, description in tasks:
        orchestrator.run_ml_task(script, description)
    
    # Print summary
    all_successful = orchestrator.print_summary()
    
    print(f"\n{'='*70}")
    if all_successful:
        print("🎉 ALL ML TASKS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  SOME ML TASKS FAILED - CHECK LOGS ABOVE")
    print(f"{'='*70}")
    print(f"🕐 Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if all_successful else 1)

if __name__ == "__main__":
    main()

