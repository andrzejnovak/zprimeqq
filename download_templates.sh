#!/bin/bash

# Download script for Z' analysis template files using xrdcp
# This script downloads ROOT template files from EOS storage

set -e

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if xrdcp is available
if ! command -v xrdcp &> /dev/null; then
    print_error "xrdcp is not available. Please ensure you have the appropriate environment setup."
    print_error "You may need to: source /cvmfs/cms.cern.ch/cmsset_default.sh"
    exit 1
fi

# Create templates directory if it doesn't exist
TEMPLATES_DIR="zprimeqq_final_templates"
if [ ! -d "$TEMPLATES_DIR" ]; then
    print_status "Creating templates directory..."
    mkdir -p "$TEMPLATES_DIR"
fi

# Define the EOS server
EOS_SERVER="root://eosproject.cern.ch/"

# Define file mappings
declare -A HIST_FILES=(
    ["2016APV"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016APV-SR-2/results/TEMPLATES_v0_0711.root"
    ["2016"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016-SR/results/TEMPLATES_v0_0711.root"
    ["2017"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2017-SR/results/TEMPLATES_v0_0711.root"
    ["2018"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2018-SR/results/TEMPLATES_v0_0711.root"
)

declare -A HIST_SIGNAL_FILES=(
    ["2016APV"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016APV-SR-2/results/TEMPLATES_v0_0711_interpolated_v0.root"
    ["2016"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2016-SR/results/TEMPLATES_v0_0711_interpolated_v0.root"
    ["2017"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2017-SR/results/TEMPLATES_v0_0711_interpolated_v0.root"
    ["2018"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/el9/9Jul24-2018-SR/results/TEMPLATES_v0_0711_interpolated_v0.root"
)

declare -A HIST_MUCR_FILES=(
    ["2016APV"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016APV-CR1/results/TEMPLATES_30May24.root"
    ["2016"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2016-CR1/results/TEMPLATES_30May24.root"
    ["2017"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2017-CR1/results/TEMPLATES_30May24.root"
    ["2018"]="/eos/project/c/contrast/public/cl/www/zprime/bamboo/19Apr24-2018-CR1/results/TEMPLATES_30May24.root"
)

# Function to download a file
download_file() {
    local year=$1
    local remote_path=$2
    local file_type=$3
    local local_filename="${year}_${file_type}.root"
    local local_path="$TEMPLATES_DIR/$local_filename"
    
    if [ -f "$local_path" ]; then
        print_warning "File $local_filename already exists. Skipping download."
        return 0
    fi
    
    print_status "Downloading $file_type templates for $year..."
    print_status "Source: $remote_path"
    print_status "Destination: $local_path"
    
    if xrdcp "$EOS_SERVER$remote_path" "$local_path"; then
        print_success "Downloaded $local_filename"
        return 0
    else
        print_error "Failed to download $local_filename"
        return 1
    fi
}

# Parse command line arguments
YEARS=("2016APV" "2016" "2017" "2018")
DOWNLOAD_ALL=true
FORCE_DOWNLOAD=false

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--year)
            YEARS=("$2")
            DOWNLOAD_ALL=false
            shift 2
            ;;
        -f|--force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -y, --year YEAR    Download files for specific year (2016APV, 2016, 2017, 2018)"
            echo "  -f, --force        Force download even if files exist"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                 # Download all years"
            echo "  $0 -y 2018         # Download only 2018 files"
            echo "  $0 -f              # Force download all files"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Remove existing files if force download is enabled
if [ "$FORCE_DOWNLOAD" = true ]; then
    print_warning "Force download enabled. Removing existing files..."
    rm -f "$TEMPLATES_DIR"/*.root
fi

print_status "Starting download of template files..."

# Download files for each requested year
for year in "${YEARS[@]}"; do
    print_status "Processing year: $year"
    
    # Download main histogram files
    download_file "$year" "${HIST_FILES[$year]}" "hist"
    
    # Download signal histogram files
    download_file "$year" "${HIST_SIGNAL_FILES[$year]}" "signal"
    
    # Download muon control region files
    download_file "$year" "${HIST_MUCR_FILES[$year]}" "mucr"
    
    echo ""
done

print_success "Download process completed!"
print_status "Files are located in: $TEMPLATES_DIR/"
print_status "Use ls -lh $TEMPLATES_DIR/ to see downloaded files"
