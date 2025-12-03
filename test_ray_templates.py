#!/usr/bin/env python3
"""
Comprehensive validation test for ALL Ray templates
Tests bart.dias generation for: map_reduce, pipeline, pool_workers, master_slave
"""

import sys
import os

# Add bart.dias to path
sys.path.insert(0, '/home/galba/Documentos/bartDias/bart.dias')

from bdias_pattern_codegen import generate_parallel_code

print("="*70)
print("BART.DIAS RAY TEMPLATES - COMPREHENSIVE VALIDATION")
print("="*70)

# Test configurations for all 4 patterns
test_cases = {
    'map_reduce': {
        'code': '''
data = list(range(1000))
results = []

for i in data:
    result = i * i + 2
    results.append(result)

total = sum(results)
''',
        'bottleneck': {
            'source': 'for i in data: result = i * i + 2',
            'type': 'for_loop',
            'lineno': 4,
            'name': 'map_reduce_loop'
        }
    },
    'pipeline': {
        'code': '''
def process_pipeline(data):
    stage1 = []
    for x in data:
        stage1.append(x * 2)
    
    stage2 = []
    for y in stage1:
        stage2.append(y + 10)
    
    return stage2
''',
        'bottleneck': {
            'source': 'def process_pipeline(data): ...',
            'type': 'function',
            'lineno': 1,
            'name': 'process_pipeline'
        }
    },
    'pool_workers': {
        'code': '''
def process_data(items):
    results = []
    for x in items:
        if x % 2 == 0:
            results.append(x * 3)
    return results
''',
        'bottleneck': {
            'source': 'def process_data(items): ...',
            'type': 'function',
            'lineno': 1,
            'name': 'process_data'
        }
    },
    'master_slave': {
        'code': '''
def worker_process(tasks):
    results = []
    for task in tasks:
        result = compute(task)
        results.append(result)
    return results
''',
        'bottleneck': {
            'source': 'def worker_process(tasks): ...',
            'type': 'function',
            'lineno': 1,
            'name': 'worker_process'
        }
    }
}

results_summary = {}

for pattern_name, config in test_cases.items():
    print(f"\n{'='*70}")
    print(f"🧪 Testing Pattern: {pattern_name.upper().replace('_', '-')}")
    print(f"{'='*70}")
    
    bottleneck = config['bottleneck'].copy()
    bottleneck['source'] = config['code']
    
    try:
        # Test Ray generation
        orig, transformed_ray, ctx = generate_parallel_code(
            bottleneck,
            pattern_name,
            ['default'],
            target_runtime='ray'
        )
        
        # Validate Ray-specific elements
        ray_checks = {
            'import ray': 'import ray' in transformed_ray,
            'ray.init': 'ray.init' in transformed_ray,
            '@ray.remote': '@ray.remote' in transformed_ray,
        }
        
        # Pattern-specific checks
        if pattern_name == 'master_slave':
            ray_checks['Worker Actor'] = 'Worker' in transformed_ray and 'class' in transformed_ray
        elif pattern_name == 'pipeline':
            ray_checks['Pipeline Actor'] = 'Actor' in transformed_ray
        
        all_passed = all(ray_checks.values())
        
        results_summary[pattern_name] = {
            'status': '✅ PASSED' if all_passed else '❌ FAILED',
            'checks': ray_checks,
            'lines': len(transformed_ray.split('\n'))
        }
        
        print(f"\n📝 Generated Code Stats:")
        print(f"   Lines: {len(transformed_ray.split('\n'))}")
        print(f"\n🔍 Validation Checks:")
        for check, result in ray_checks.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check}: {result}")
        
        if all_passed:
            print(f"\n✨ {pattern_name.upper()} passed all checks!")
        else:
            print(f"\n⚠️  {pattern_name.upper()} failed some checks")
            print("\nGenerated code preview (first 15 lines):")
            for i, line in enumerate(transformed_ray.split('\n')[:15], 1):
                print(f"   {i:2d}: {line}")
        
    except Exception as e:
        results_summary[pattern_name] = {
            'status': '❌ ERROR',
            'error': str(e)
        }
        print(f"\n❌ Error testing {pattern_name}: {e}")
        import traceback
        traceback.print_exc()

# Final Summary
print("\n" + "="*70)
print("📊 FINAL SUMMARY")
print("="*70)

for pattern, result in results_summary.items():
    print(f"\n{pattern.upper().replace('_', '-'):20s} {result['status']}")
    if 'checks' in result:
        passed = sum(1 for v in result['checks'].values() if v)
        total = len(result['checks'])
        print(f"{'':20s} Checks: {passed}/{total} passed")
        print(f"{'':20s} Code: {result['lines']} lines")

all_patterns_passed = all(
    result['status'] == '✅ PASSED' 
    for result in results_summary.values()
)

print("\n" + "="*70)
if all_patterns_passed:
    print("🎉 ALL 4 PATTERNS VALIDATED SUCCESSFULLY!")
    print("✅ map_reduce, pipeline, pool_workers, master_slave")
else:
    print("⚠️  Some patterns failed validation - review output above")
print("="*70)

sys.exit(0 if all_patterns_passed else 1)
