#!/bin/bash
# Build script for LEAN4 proofs

set -e

echo "Building LEAN4 proofs for AAOS..."

# Ensure we're in the lean4 directory
cd "$(dirname "$0")"

# Update dependencies
echo "Updating dependencies..."
lake update

# Build all modules
echo "Building all proof modules..."
lake build

# Run specific tests if requested
if [ "$1" = "test" ]; then
    echo "Running proof verification tests..."
    lake build AAOSProofs.Test
    echo "Test build successful!"
fi

echo "LEAN4 build completed successfully!"