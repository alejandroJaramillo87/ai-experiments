#!/usr/bin/env python3
"""
Systematic Base Calibration CLI Script

This script provides a command-line interface to the SystematicBaseCalibrator
for progressive domain calibration (easy → medium → hard).

Usage:
    python scripts/calibration/systematic_base_calibration.py [domain]
    
If no domain specified, calibrates all core domains in sequence.
"""

import sys
import os
from pathlib import Path

# Add core modules to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

try:
    from core.calibration_engine import SystematicBaseCalibrator, SystematicCalibrationResult
except ImportError as e:
    print(f"Error importing calibration engine: {e}")
    print("Please ensure core/calibration_engine.py is available")
    sys.exit(1)

def main():
    """Main entry point for systematic base calibration"""
    
    # Core domains for systematic calibration
    core_domains = [
        "reasoning", "creativity", "language", 
        "social", "knowledge", "integration"
    ]
    
    # Parse command line arguments
    domain = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        # Initialize the calibrator
        calibrator = SystematicBaseCalibrator()
        
        if domain:
            # Calibrate specific domain
            print(f"Starting systematic calibration for domain: {domain}")
            result = calibrator.calibrate_domain_progression(domain)
            
            # Report results
            print(f"\nCalibration Results for {domain}:")
            print(f"Status: {result.status}")
            print(f"Overall Score: {result.overall_score:.2f}")
            
            if hasattr(result, 'detailed_results'):
                for difficulty, score in result.detailed_results.items():
                    print(f"  {difficulty}: {score:.2f}")
        
        else:
            # Calibrate all core domains
            print("Starting systematic calibration for all core domains")
            print(f"Domains: {', '.join(core_domains)}")
            
            all_results = {}
            overall_success = True
            
            for domain_name in core_domains:
                print(f"\n{'='*50}")
                print(f"Calibrating Domain: {domain_name.upper()}")
                print(f"{'='*50}")
                
                try:
                    result = calibrator.calibrate_domain_progression(domain_name)
                    all_results[domain_name] = result
                    
                    print(f"Status: {result.status}")
                    print(f"Overall Score: {result.overall_score:.2f}")
                    
                    # Check if calibration failed and should halt progression
                    if hasattr(result, 'should_halt') and result.should_halt:
                        print(f"⚠️  Calibration halted for {domain_name} - fixing required")
                        overall_success = False
                        break
                        
                except Exception as e:
                    print(f"❌ Error calibrating {domain_name}: {e}")
                    overall_success = False
                    break
            
            # Final summary
            print(f"\n{'='*60}")
            print("SYSTEMATIC CALIBRATION SUMMARY")
            print(f"{'='*60}")
            
            for domain_name, result in all_results.items():
                status_emoji = "✅" if result.status == "success" else "❌"
                print(f"{status_emoji} {domain_name}: {result.overall_score:.2f} ({result.status})")
            
            if overall_success:
                print("\n🎯 All domains successfully calibrated!")
                print("System ready for production deployment.")
            else:
                print("\n⚠️  Some domains require attention before production.")
                
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()