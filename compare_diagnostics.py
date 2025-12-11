#!/usr/bin/env python3
"""
Compare R and Python calibration diagnostics stage by stage
"""

import re
import numpy as np

def parse_diagnostics(filename):
    """Parse diagnostic output from R or Python"""
    with open(filename, 'r') as f:
        content = f.read()
    
    diagnostics = {}
    
    # Parse initial parameters
    match = re.search(r'=== DIAGNOSTIC: INITIAL PARAMETERS.*?===\s*\n(.*?)(?=\n\s*Step|\n===|$)', content, re.DOTALL)
    if match:
        init_text = match.group(1)
        diagnostics['initial'] = {}
        for param in ['Lb', 'X1', 'Wb', 'POR', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz', 'Error', 'BFF']:
            match_param = re.search(rf'{param}=([\d.]+)', init_text)
            if match_param:
                diagnostics['initial'][param] = float(match_param.group(1))
    
    # Parse Step 1 - after cal_initial
    match = re.search(r'=== DIAGNOSTIC: STEP 1 - AFTER cal_initial ===\s*\n.*?iterations: (\d+).*?final objective: ([\d.-]+)\s*\n.*?Optimized params \(linear\): Lb=([\d.]+), Wb=([\d.]+), ALPHA=([\d.]+), Ks=([\d.]+), Kb=([\d.]+), Kz=([\d.]+)', content, re.DOTALL)
    if match:
        diagnostics['step1_cal_initial'] = {
            'iterations': int(match.group(1)),
            'objective': float(match.group(2)),
            'Lb': float(match.group(3)),
            'Wb': float(match.group(4)),
            'ALPHA': float(match.group(5)),
            'Ks': float(match.group(6)),
            'Kb': float(match.group(7)),
            'Kz': float(match.group(8))
        }
    
    # Parse Step 1 - after cal_surface
    match = re.search(r'=== DIAGNOSTIC: STEP 1 - AFTER cal_surface ===\s*\n.*?iterations: (\d+).*?final objective: ([\d.-]+)\s*\n.*?Optimized params \(linear\): Lb=([\d.]+), Wb=([\d.]+), ALPHA=([\d.]+), Ks=([\d.]+)', content, re.DOTALL)
    if match:
        diagnostics['step1_cal_surface'] = {
            'iterations': int(match.group(1)),
            'objective': float(match.group(2)),
            'Lb': float(match.group(3)),
            'Wb': float(match.group(4)),
            'ALPHA': float(match.group(5)),
            'Ks': float(match.group(6))
        }
    
    # Parse Step 1 - after final BFS run
    match = re.search(r'=== DIAGNOSTIC: STEP 1 - AFTER final BFS run ===\s*\n(.*?)(?=\n\s*Step|\n===|$)', content, re.DOTALL)
    if match:
        step1_text = match.group(1)
        diagnostics['step1_final'] = {}
        for param in ['Lb', 'X1', 'Wb', 'POR', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz', 'Error', 'BFF']:
            match_param = re.search(rf'{param}=([\d.]+)', step1_text)
            if match_param:
                diagnostics['step1_final'][param] = float(match_param.group(1))
    
    # Parse Step 2 - after cal_base
    match = re.search(r'=== DIAGNOSTIC: STEP 2 - AFTER cal_base.*?===\s*\n.*?iterations: (\d+).*?final objective: ([\d.-]+)\s*\n.*?Optimized params: Lb=([\d.]+), Wb=([\d.]+), Kb=([\d.]+), Kz=([\d.]+)', content, re.DOTALL)
    if match:
        diagnostics['step2_cal_base'] = {
            'iterations': int(match.group(1)),
            'objective': float(match.group(2)),
            'Lb': float(match.group(3)),
            'Wb': float(match.group(4)),
            'Kb': float(match.group(5)),
            'Kz': float(match.group(6))
        }
    
    # Parse Step 2 - after cal_surface
    match = re.search(r'=== DIAGNOSTIC: STEP 2 - AFTER cal_surface.*?===\s*\n.*?iterations: (\d+).*?final objective: ([\d.-]+)\s*\n.*?Optimized params \(linear\): Lb=([\d.]+), Wb=([\d.]+), ALPHA=([\d.]+), Ks=([\d.]+)', content, re.DOTALL)
    if match:
        diagnostics['step2_cal_surface'] = {
            'iterations': int(match.group(1)),
            'objective': float(match.group(2)),
            'Lb': float(match.group(3)),
            'Wb': float(match.group(4)),
            'ALPHA': float(match.group(5)),
            'Ks': float(match.group(6))
        }
    
    # Parse Step 3 - after cal_initial
    match = re.search(r'=== DIAGNOSTIC: STEP 3 - AFTER cal_initial ===\s*\n.*?iterations: (\d+).*?final objective: ([\d.-]+)\s*\n.*?Optimized params \(linear\): Lb=([\d.]+), Wb=([\d.]+), ALPHA=([\d.]+), Ks=([\d.]+), Kb=([\d.]+), Kz=([\d.]+)', content, re.DOTALL)
    if match:
        diagnostics['step3_cal_initial'] = {
            'iterations': int(match.group(1)),
            'objective': float(match.group(2)),
            'Lb': float(match.group(3)),
            'Wb': float(match.group(4)),
            'ALPHA': float(match.group(5)),
            'Ks': float(match.group(6)),
            'Kb': float(match.group(7)),
            'Kz': float(match.group(8))
        }
    
    # Parse final parameters
    match = re.search(r'=== DIAGNOSTIC: FINAL PARAMETERS.*?===\s*\n(.*?)(?=\n\s*===|$)', content, re.DOTALL)
    if match:
        final_text = match.group(1)
        diagnostics['final'] = {}
        for param in ['Lb', 'X1', 'Wb', 'POR', 'ALPHA', 'BETA', 'Ks', 'Kb', 'Kz', 'Error', 'BFF']:
            match_param = re.search(rf'{param}=([\d.]+)', final_text)
            if match_param:
                diagnostics['final'][param] = float(match_param.group(1))
    
    # Parse beta selection
    match = re.search(r'Selected best beta=([\d.]+)', content)
    if match:
        diagnostics['best_beta'] = float(match.group(1))
    
    match = re.search(r'Selected row \d+: BETA=([\d.]+)', content)
    if match:
        diagnostics['selected_beta'] = float(match.group(1))
    
    return diagnostics

def compare_stage(r_diag, py_diag, stage_name):
    """Compare a specific stage between R and Python"""
    if stage_name not in r_diag or stage_name not in py_diag:
        return None
    
    r_vals = r_diag[stage_name]
    py_vals = py_diag[stage_name]
    
    common_keys = set(r_vals.keys()) & set(py_vals.keys())
    if not common_keys:
        return None
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: {stage_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    print(f"{'Parameter':<15} {'R Value':<20} {'Python Value':<20} {'Difference':<20} {'% Diff':<15}")
    print("-" * 90)
    
    differences = []
    for key in sorted(common_keys):
        r_val = r_vals[key]
        py_val = py_vals[key]
        diff = py_val - r_val
        if r_val != 0:
            pct_diff = (diff / r_val) * 100
        else:
            pct_diff = 0.0 if diff == 0 else float('inf')
        
        differences.append((key, abs(pct_diff)))
        
        print(f"{key:<15} {r_val:<20.6f} {py_val:<20.6f} {diff:<20.6f} {pct_diff:<15.2f}%")
    
    if differences:
        max_diff_param, max_diff = max(differences, key=lambda x: x[1])
        print(f"\nMaximum difference: {max_diff_param} ({max_diff:.2f}%)")
    
    return differences

def main():
    # Parse R diagnostics
    r_file = 'bfs/calibration_R_original/r_diagnostics_output.txt'
    try:
        r_diag = parse_diagnostics(r_file)
        print(f"✓ Parsed R diagnostics from {r_file}")
    except Exception as e:
        print(f"✗ Error parsing R diagnostics: {e}")
        return
    
    # Parse Python diagnostics (from terminal output)
    # We'll need to capture Python output to a file first
    print("\nNote: Python diagnostics need to be saved to a file.")
    print("Run: python main_calib2.py > python_diagnostics_output.txt 2>&1")
    print("Then run this script again.\n")
    
    py_file = 'python_diagnostics_output.txt'
    try:
        py_diag = parse_diagnostics(py_file)
        print(f"✓ Parsed Python diagnostics from {py_file}")
    except Exception as e:
        print(f"✗ Error parsing Python diagnostics: {e}")
        print("Please run: python main_calib2.py > python_diagnostics_output.txt 2>&1")
        return
    
    # Compare each stage
    stages = [
        'initial',
        'step1_cal_initial',
        'step1_cal_surface',
        'step1_final',
        'step2_cal_base',
        'step2_cal_surface',
        'step3_cal_initial',
        'final'
    ]
    
    for stage in stages:
        compare_stage(r_diag, py_diag, stage)
    
    # Compare beta selection
    if 'best_beta' in r_diag and 'best_beta' in py_diag:
        print(f"\n{'='*80}")
        print("BETA SELECTION COMPARISON")
        print(f"{'='*80}")
        print(f"R best beta: {r_diag['best_beta']}")
        print(f"Python best beta: {py_diag['best_beta']}")
        print(f"Difference: {abs(py_diag['best_beta'] - r_diag['best_beta'])}")
    
    if 'selected_beta' in r_diag and 'selected_beta' in py_diag:
        print(f"\nR selected beta: {r_diag['selected_beta']}")
        print(f"Python selected beta: {py_diag['selected_beta']}")
        print(f"Difference: {abs(py_diag['selected_beta'] - r_diag['selected_beta'])}")

if __name__ == '__main__':
    main()

