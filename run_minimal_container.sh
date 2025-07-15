#!/bin/bash

# Script to run the minimal combine+rhalphalib container
set -e

IMAGE_NAME="combine-rhalphalib:root6.22-py3.8-rhal0.3.0-combine9.2.1"
docker build -f Dockerfile.minimal -t "$IMAGE_NAME" .
echo "Starting minimal combine container with rhalphalib..."

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "Error: Image $IMAGE_NAME not found!"
    echo "Please build it first with: docker build -f Dockerfile.minimal -t $IMAGE_NAME ."
    exit 1
fi

# Run the container with volume mounting
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    echo "Running in interactive mode..."
    docker run --rm -it -v "$(pwd)":/analysis/data "$IMAGE_NAME" bash
elif [ "$1" = "--test" ]; then
    echo "Testing container functionality..."
    echo "Testing combine:"
    docker run --rm "$IMAGE_NAME" combine --help | head -3
    echo ""
    echo "Testing rhalphalib:"
    docker run --rm "$IMAGE_NAME" python3 -c "import rhalphalib; print('✓ rhalphalib working')"
    echo ""
    echo "Testing Python packages:"
    docker run --rm "$IMAGE_NAME" python3 -c "import numpy, pandas, matplotlib; print('✓ Basic packages working')"
    echo ""
    echo "All tests passed! Container is ready to use."
else
    echo "Usage:"
    echo "  $0 --interactive   # Run container interactively with volume mounting"
    echo "  $0 --test          # Test container functionality"
    echo ""
    echo "Container info:"
    echo "  - Base: CERN combine v9.2.1-slim"
    echo "  - Includes: combine, rhalphalib v0.3.0"
    echo "  - Size: ~5.3GB (much smaller than custom builds)"
fi
