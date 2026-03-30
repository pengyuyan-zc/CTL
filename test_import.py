#!/usr/bin/env python3
"""
Test script to verify federated_ctn_lt module import
"""

print("Testing federated_ctn_lt module import...")
try:
    import federated_ctn_lt
    print("✓ federated_ctn_lt module imported successfully")
    
    # Test importing submodules
    try:
        from federated_ctn_lt.models import CTN_LT
        print("✓ CTN_LT model imported successfully")
    except Exception as e:
        print(f"✗ Failed to import CTN_LT: {e}")
    
    try:
        from federated_ctn_lt.federated import FederatedServer, FederatedClient
        print("✓ FederatedServer and FederatedClient imported successfully")
    except Exception as e:
        print(f"✗ Failed to import federated modules: {e}")
    
    try:
        from federated_ctn_lt.data import FederatedDataPartitioner
        print("✓ FederatedDataPartitioner imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data modules: {e}")
    
    try:
        from federated_ctn_lt.evaluation import MetricsCalculator
        print("✓ MetricsCalculator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import evaluation modules: {e}")
        
except Exception as e:
    print(f"✗ Failed to import federated_ctn_lt: {e}")

print("Import test completed!")
