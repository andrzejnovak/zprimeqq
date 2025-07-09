#!/bin/bash
# Script to download template files from EOS to local directory
# Usage: ./download_files.sh [output_directory]

set -e

# Default output directory
OUTPUT_DIR="${1:-./templates}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Downloading template files to: $OUTPUT_DIR"
echo "================================================"

# Function to download a file
download_file() {
    local remote_path="$1"
    local filename="$2"
    local local_path="$OUTPUT_DIR/$filename"
    
    echo "Downloading: $filename"
    
    # Try xrdcp first, fall back to eos cp
    if command -v xrdcp &> /dev/null; then
        xrootd_url="root://eosproject.cern.ch/$remote_path"
        if xrdcp "$xrootd_url" "$local_path"; then
            echo "✓ Downloaded: $filename"
        else
            echo "✗ Failed to download: $filename"
        fi
    elif command -v eos &> /dev/null; then
        if eos cp "$remote_path" "$local_path"; then
            echo "✓ Downloaded: $filename"
        else
            echo "✗ Failed to download: $filename"
        fi
    else
        echo "✗ Neither xrdcp nor eos command available"
        exit 1
    fi
}

# Main template files
echo "Downloading main template files..."
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016APV-SR-2/results/TEMPLATES_v0_0711.root" "TEMPLATES_v0_0711_2016APV.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016-SR/results/TEMPLATES_v0_0711.root" "TEMPLATES_v0_0711_2016.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2017-SR/results/TEMPLATES_v0_0711.root" "TEMPLATES_v0_0711_2017.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2018-SR/results/TEMPLATES_v0_0711.root" "TEMPLATES_v0_0711_2018.root"

# Interpolated signal template files
echo -e "\nDownloading interpolated signal template files..."
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016APV-SR-2/results/TEMPLATES_v0_0711_interpolated_v0.root" "TEMPLATES_v0_0711_interpolated_v0_2016APV.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016-SR/results/TEMPLATES_v0_0711_interpolated_v0.root" "TEMPLATES_v0_0711_interpolated_v0_2016.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2017-SR/results/TEMPLATES_v0_0711_interpolated_v0.root" "TEMPLATES_v0_0711_interpolated_v0_2017.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2018-SR/results/TEMPLATES_v0_0711_interpolated_v0.root" "TEMPLATES_v0_0711_interpolated_v0_2018.root"

# Muon control region template files
echo -e "\nDownloading muon control region template files..."
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016APV-CR1/results/TEMPLATES_30May24.root" "TEMPLATES_30May24_2016APV_mucr.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016-CR1/results/TEMPLATES_30May24.root" "TEMPLATES_30May24_2016_mucr.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2017-CR1/results/TEMPLATES_30May24.root" "TEMPLATES_30May24_2017_mucr.root"
download_file "/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2018-CR1/results/TEMPLATES_30May24.root" "TEMPLATES_30May24_2018_mucr.root"


echo -e "\nDownload completed!"
echo "Files downloaded to: $OUTPUT_DIR"

# Create a README file
cat > "$OUTPUT_DIR/README.md" << EOF
# Downloaded Template Files

This directory contains template files downloaded from EOS for the zprime analysis.

## File descriptions:

### Main template files (TEMPLATES_v0_0711_*.root):
- Background and signal templates for signal region analysis
- One file per year: 2016APV, 2016, 2017, 2018

### Interpolated signal files (TEMPLATES_v0_0711_interpolated_v0_*.root):
- Signal templates with mass interpolation
- Used for limit setting across different mass points

### Muon control region files (TEMPLATES_30May24_*_mucr.root):
- Templates from muon control region
- Used for background estimation and validation

## Usage:
Update the file paths in your analysis scripts to point to these local files instead of the EOS paths.

Downloaded on: $(date)
EOF

echo "Created README.md with file descriptions"
